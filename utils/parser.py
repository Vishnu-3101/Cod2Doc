## Changes from parser.py
"""
ImportCollector class:
    __init__()
    visit_Import
    visit_ImportFrom()

DependencyCollector class:
    __init__()
    visit_ClassDef
    visit_Call
    _resolve_name_from_from_imports - added
    _process_attribute

Dependency_parser class:
    __init__()
    parse_repository


"""


# Copyright (c) Meta Platforms, Inc. and affiliates
"""
AST-based Python code parser that extracts dependency information between code components.

Improved import resolution for repo-relative modules and alias handling.
"""

import ast
import os
import json
import logging
import builtins
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Built-in Python types and modules that should be excluded from dependencies
BUILTIN_TYPES = {name for name in dir(builtins)}
STANDARD_MODULES = {
    'abc', 'argparse', 'array', 'asyncio', 'base64', 'collections', 'copy',
    'csv', 'datetime', 'enum', 'functools', 'glob', 'io', 'itertools',
    'json', 'logging', 'math', 'os', 'pathlib', 'random', 're', 'shutil',
    'string', 'sys', 'time', 'typing', 'uuid', 'warnings', 'xml'
}
EXCLUDED_NAMES = {'self', 'cls'}


@dataclass
class CodeComponent:
    """
    Represents a single code component (function, class, method, or assignment) in a Python codebase.
    """
    id: str
    node: ast.AST
    component_type: str  # 'class', 'function', 'method', 'assignment'
    file_path: str
    relative_path: str
    depends_on: Set[str] = field(default_factory=set)
    source_code: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    has_docstring: bool = False
    docstring: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'component_type': self.component_type,
            'file_path': self.file_path,
            'relative_path': self.relative_path,
            'depends_on': list(self.depends_on),
            'source_code': self.source_code,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'has_docstring': self.has_docstring,
            'docstring': self.docstring
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'CodeComponent':
        component = CodeComponent(
            id=data['id'],
            node=None,
            component_type=data['component_type'],
            file_path=data['file_path'],
            relative_path=data['relative_path'],
            depends_on=set(data.get('depends_on', [])),
            source_code=data.get('source_code'),
            start_line=data.get('start_line', 0),
            end_line=data.get('end_line', 0),
            has_docstring=data.get('has_docstring', False),
            docstring=data.get('docstring', "")
        )
        return component


class ImportCollector(ast.NodeVisitor):
    """
    Collects imports and resolves them to repo-relative module paths when possible.

    Attributes produced:
      - imports: dict mapping identifier used in code -> full module path (e.g. 'utils' -> 'AutoDiff.utils')
      - from_imports: dict mapping resolved_module (full) -> set(imported_names)
    """

    def __init__(self, current_module: str, repo_modules: Set[str]):
        self.current_module = current_module  # e.g., "AutoDiff.main"
        self.repo_modules = repo_modules
        self.imports: Dict[str, str] = {}      # identifier -> module_full_path
        self.from_imports: Dict[str, Set[str]] = {}  # resolved_module -> set(names)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            full_name = alias.name  # e.g. "AutoDiff.utils" or "os"
            if alias.asname:
                identifier = alias.asname
                self.imports[identifier] = full_name
            else:
                # binding is top-level package name (first part)
                identifier = full_name.split('.')[0]
                # store mapping identifier -> full_name (we'll use logic in DependencyCollector)
                self.imports[identifier] = full_name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """
        Resolve 'from X import Y' with support for:
          - absolute imports
          - implicit repo-relative resolution (prefix with current package)
          - relative imports (node.level)
        """
        module = node.module  # may be None for 'from . import X'
        level = getattr(node, "level", 0)

        resolved_module = None

        # Build current module parts (e.g., AutoDiff.main -> ['AutoDiff', 'main'])
        cur_parts = self.current_module.split('.')

        # Handle explicit relative imports (node.level > 0)
        if level and level > 0:
            # Compute base by going up 'level' directories from current module's package
            # drop last element(s) for level: typical: from .utils import X -> level=1
            base_parts = cur_parts[:-level]
            if module:
                base_parts = base_parts + module.split('.')
            if base_parts:
                candidate = '.'.join(base_parts)
                if candidate in self.repo_modules:
                    resolved_module = candidate
                else:
                    resolved_module = candidate  # keep candidate even if not in repo (external or unresolved)
            else:
                resolved_module = module or ''
        else:
            # Absolute import or plain 'from utils import X'
            if module:
                # If exact module exists in repo, use it
                if module in self.repo_modules:
                    resolved_module = module
                else:
                    # Try to resolve by prefixing parent packages of current_module:
                    parent_parts = cur_parts[:-1]  # module's package
                    resolved = None
                    while parent_parts:
                        cand = '.'.join(parent_parts + module.split('.'))
                        if cand in self.repo_modules:
                            resolved = cand
                            break
                        parent_parts = parent_parts[:-1]
                    if resolved:
                        resolved_module = resolved
                    else:
                        # Keep module text (external or unresolved)
                        resolved_module = module
            else:
                # 'from . import X' or 'from  import X' without module, treat as parent package
                parent_candidate = '.'.join(cur_parts[:-1])
                if parent_candidate in self.repo_modules:
                    resolved_module = parent_candidate
                else:
                    resolved_module = parent_candidate  # external/unresolved parent

        # Record imported names under the resolved module key
        key = resolved_module or (module or "")
        if key not in self.from_imports:
            self.from_imports[key] = set()

        for alias in node.names:
            if alias.name == '*':
                # star imports are hard to resolve; record wildcard marker
                # keep '*' as name so downstream can detect
                self.from_imports[key].add('*')
            else:
                self.from_imports[key].add(alias.asname or alias.name)

        self.generic_visit(node)


class MethodDependencyCollector(ast.NodeVisitor):
    """
    For methods: capture self.xxx accesses and map to other methods if they exist.
    """
    def __init__(self, class_id: str, method_id: str, class_methods: Dict[str, str]):
        self.class_id = class_id
        self.method_id = method_id
        self.class_methods = class_methods
        self.self_attr_refs = set()

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == 'self' and isinstance(node.ctx, ast.Load):
            self.self_attr_refs.add(node.attr)
        self.generic_visit(node)

    def get_method_dependencies(self) -> Set[str]:
        deps = set()
        for attr in self.self_attr_refs:
            if attr in self.class_methods:
                deps.add(self.class_methods[attr])
        return deps


class DependencyCollector(ast.NodeVisitor):
    """
    Collects dependencies between code components by analyzing attribute access,
    function calls, class bases and name references. It uses the import mappings
    resolved by ImportCollector to link to repo modules.
    """

    def __init__(self, imports: Dict[str, str], from_imports: Dict[str, Set[str]], current_module: str, repo_modules: Set[str]):
        # imports: identifier -> module_full_path (e.g. 'utils' -> 'AutoDiff.utils' or 'AutoDiff' -> 'AutoDiff.utils')
        # from_imports: resolved_module -> set(names)
        self.imports = imports
        self.from_imports = from_imports
        self.current_module = current_module
        self.repo_modules = repo_modules
        self.dependencies: Set[str] = set()
        self._current_class = None
        self.local_variables: Set[str] = set()

    def visit_ClassDef(self, node: ast.ClassDef):
        old = self._current_class
        self._current_class = node.name
        for base in node.bases:
            if isinstance(base, ast.Name):
                # # If base is imported via from_imports, map to module.Class
                # mapped = self._resolve_name_from_from_imports(base.id)
                # if mapped:
                #     self.dependencies.add(mapped)
                # else:
                    # fallback: local module reference
                self._add_dependency(base.id)
            elif isinstance(base, ast.Attribute):
                self._process_attribute(base)
        self.generic_visit(node)
        self._current_class = old

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_variables.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # handle direct function calls possibly imported via 'from module import func'
        if isinstance(node.func, ast.Name):
            name = node.func.id
            # # check from_imports mapping
            # mapped = self._resolve_name_from_from_imports(name)
            # if mapped:
            #     self.dependencies.add(mapped)
            # else:
            self._add_dependency(name)
        elif isinstance(node.func, ast.Attribute):
            self._process_attribute(node.func)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self._add_dependency(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        self._process_attribute(node)
        self.generic_visit(node)

    # def _resolve_name_from_from_imports(self, name: str) -> Optional[str]:
    #     """
    #     If `name` was imported via `from <module> import name`, return '<module>.<name>'
    #     where <module> is the resolved module key (prefer repo modules keys).
    #     """
    #     for module_key, names in self.from_imports.items():
    #         if name in names:
    #             # If module_key is a resolved repo module, use it; otherwise we still return module_key
    #             if module_key:
    #                 return f"{module_key}.{name}"
    #     return None

    def _process_attribute(self, node: ast.Attribute):
        """
        Handles dotted expressions and resolves them to repo modules where possible.
        Example patterns:
          - alias.Class -> import alias maps to AutoDiff.utils -> dependency AutoDiff.utils.Class
          - AutoDiff.utils.Class -> longest prefix match of parts (AutoDiff.utils) found in repo_modules
          - from-import based attribute usage -> resolved via from_imports
        """
        # parts = []
        # current = node
        # while isinstance(current, ast.Attribute):
        #     parts.insert(0, current.attr)
        #     current = current.value

        # if isinstance(current, ast.Name):
        #     parts.insert(0, current.id)
        # else:
        #     # not a simple dotted name (could be call result etc.)
        #     return

        # # skip local variables and excluded names
        # if parts[0] in self.local_variables or parts[0] in EXCLUDED_NAMES:
        #     return

        # # 1) If first name is an alias from 'import ... as ...' or 'import ...'
        # if parts[0] in self.imports:
        #     module_full = self.imports[parts[0]]  # e.g. 'AutoDiff.utils' or 'os' or 'AutoDiff'
        #     module_full_parts = module_full.split('.')
        #     # If the module_full parts appear at the start of the dotted name (e.g. 'AutoDiff.utils.Class')
        #     # find how many module parts match the `parts` prefix
        #     m = 0
        #     for i in range(min(len(module_full_parts), len(parts))):
        #         if parts[i] == module_full_parts[i]:
        #             m += 1
        #         else:
        #             break

        #     if m == len(module_full_parts):
        #         # matched full module path already present in code (e.g. 'AutoDiff.utils...')
        #         remainder_index = m
        #     elif parts[0] == parts[0]:  # alias usage case: alias maps to module_full -> use alias form
        #         # alias used as a shorthand (e.g., 'utils.Class' while imports['utils'] == 'AutoDiff.utils')
        #         remainder_index = 1
        #     else:
        #         remainder_index = len(module_full_parts)

        #     # Determine the attribute name referenced after the module path
        #     if remainder_index < len(parts):
        #         attr_name = parts[remainder_index]
        #         candidate_module = module_full  # resolved module path
        #         # skip stdlib
        #         top_module = candidate_module.split('.')[0]
        #         if top_module in STANDARD_MODULES:
        #             return
        #         # only add if candidate_module is in repo or if import likely resolved
        #         if candidate_module in self.repo_modules or candidate_module:
        #             self.dependencies.add(f"{candidate_module}.{attr_name}")
        #     else:
        #         # attribute refers directly to the module imported without further attribute (rare)
        #         # we don't add dependency for bare module reference
        #         return

        #     return

        # # 2) Not an alias: find the longest prefix of parts that matches a repo module
        # #    e.g., parts = ['AutoDiff','utils','Class'] => look for 'AutoDiff.utils' or 'AutoDiff'
        # for k in range(len(parts), 0, -1):
        #     candidate_module = '.'.join(parts[:k])
        #     if candidate_module in self.repo_modules:
        #         # attribute after the module (if any)
        #         if k < len(parts):
        #             attr = parts[k]
        #             # skip stdlib top-level
        #             if candidate_module.split('.')[0] in STANDARD_MODULES:
        #                 return
        #             self.dependencies.add(f"{candidate_module}.{attr}")
        #         return

        # # 3) Check if the dotted expression is of the form 'module.Name' matched in from_imports keys
        # #    e.g., if 'utils' was recorded in from_imports as resolved to 'AutoDiff.utils', handle that.
        # if parts[0] in self.from_imports:
        #     # parts[0] refers to a resolved module key in from_imports (rare because keys are resolved module strings)
        #     if len(parts) > 1 and parts[1] in self.from_imports[parts[0]]:
        #         self.dependencies.add(f"{parts[0]}.{parts[1]}")
        #     return
        

        parts = []
        current = node
        
        # Traverse the attribute chain (e.g., module.submodule.Class.method)
        while isinstance(current, ast.Attribute):
            parts.insert(0, current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.insert(0, current.id)
            
            # Skip if the first part is a local variable
            if parts[0] in self.local_variables:
                return
                
            # Skip if the first part is in our excluded names
            if parts[0] in EXCLUDED_NAMES:
                return
                
            # Check if the first part is an imported module
            if parts[0] in self.imports:
                module_path = parts[0]
                # Skip standard library modules
                if module_path in STANDARD_MODULES:
                    return
                    
                # If it's a repo module, add as dependency
                if module_path in self.repo_modules:
                    if len(parts) > 1:
                        # Example: module.Class or module.function
                        self.dependencies.add(f"{module_path}.{parts[1]}")
            
            # Check from imports
            elif parts[0] in self.from_imports.keys():
                # Skip standard library modules
                if parts[0] in STANDARD_MODULES:
                    return
                    
                # Check if the name is in the imported names
                if len(parts) > 1 and parts[1] in self.from_imports[parts[0]]:
                    self.dependencies.add(f"{parts[0]}.{parts[1]}")

    def _add_dependency(self, name: str):
        """
        Add a dependency for a simple name reference:
           - If it's a built-in or local var, ignore
           - If name was imported via 'from module import name', map to module.name
           - Otherwise assume local module reference current_module.name
        """
        if name in BUILTIN_TYPES or name in EXCLUDED_NAMES or name in self.local_variables:
            return

        # Check from_imports first
        for module, imported_names in self.from_imports.items():
            # Skip standard library modules
            if module in STANDARD_MODULES:
                continue
                
            if name in imported_names and module in self.repo_modules:
                self.dependencies.add(f"{module}.{name}")
                return

        # mapped = self._resolve_name_from_from_imports(name)
        # if mapped:
        #     self.dependencies.add(mapped)
        #     return

        # Fallback: local module component reference
        local_component_id = f"{self.current_module}.{name}"
        self.dependencies.add(local_component_id)


def add_parent_to_nodes(tree: ast.AST) -> None:
    """
    Add a 'parent' attribute to each node in the AST for upward navigation.
    """
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node


class DependencyParser:
    """
    Parses Python code to build a dependency graph between code components.
    """

    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.components: Dict[str, CodeComponent] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.modules: Set[str] = set()

    def parse_repository(self):
        logger.info(f"Parsing repository at {self.repo_path}")

        # First pass: collect modules and code components
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.repo_path)
                module_path = self._file_to_module_path(relative_path)
                self.modules.add(module_path)

        # Second pass: parse files (now that self.modules is populated)
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.repo_path)
                module_path = self._file_to_module_path(relative_path)
                self._parse_file(file_path, relative_path, module_path)

        # Resolve dependencies (analyze component bodies)
        self._resolve_dependencies()

        # Add method dependencies to classes
        self._add_class_method_dependencies()

        logger.info(f"Found {len(self.components)} code components")
        return self.components

    def _file_to_module_path(self, file_path: str) -> str:
        """Convert a file path (relative to repo) to a Python module path (dotted)."""
        path = file_path[:-3] if file_path.endswith(".py") else file_path
        return path.replace(os.path.sep, ".")

    def _parse_file(self, file_path: str, relative_path: str, module_path: str):
        """Parse a single Python file to collect code components."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)
            add_parent_to_nodes(tree)

            # Collect imports with repo-aware resolution
            import_collector = ImportCollector(module_path, self.modules)
            import_collector.visit(tree)

            # Collect code components
            self._collect_components(tree, file_path, relative_path, module_path, source)

        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Error parsing {file_path}: {e}")

    def _collect_components(self, tree: ast.AST, file_path: str, relative_path: str,
                            module_path: str, source: str):
        """Collect classes, top-level functions, methods and assignments (top-level)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_id = f"{module_path}.{node.name}"
                has_docstring = (
                    len(node.body) > 0
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                )
                docstring = self._get_docstring(source, node) if has_docstring else ""
                component = CodeComponent(
                    id=class_id,
                    node=node,
                    component_type="class",
                    file_path=file_path,
                    relative_path=relative_path,
                    source_code=f"class {class_id.split('.')[-1]}",
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    has_docstring=has_docstring,
                    docstring=docstring
                )
                self.components[class_id] = component

                # methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_id = f"{class_id}.{item.name}"
                        method_has_docstring = (
                            len(item.body) > 0
                            and isinstance(item.body[0], ast.Expr)
                            and isinstance(item.body[0].value, ast.Constant)
                            and isinstance(item.body[0].value.value, str)
                        )
                        method_docstring = self._get_docstring(source, item) if method_has_docstring else ""
                        method_component = CodeComponent(
                            id=method_id,
                            node=item,
                            component_type="method",
                            file_path=file_path,
                            relative_path=relative_path,
                            source_code=self._get_source_segment(source, item),
                            start_line=item.lineno,
                            end_line=getattr(item, "end_lineno", item.lineno),
                            has_docstring=method_has_docstring,
                            docstring=method_docstring
                        )
                        self.components[method_id] = method_component

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only collect top-level functions
                if hasattr(node, 'parent') and isinstance(node.parent, ast.Module):
                    func_id = f"{module_path}.{node.name}"
                    has_docstring = (
                        len(node.body) > 0
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    )
                    docstring = self._get_docstring(source, node) if has_docstring else ""
                    component = CodeComponent(
                        id=func_id,
                        node=node,
                        component_type="function",
                        file_path=file_path,
                        relative_path=relative_path,
                        source_code=self._get_source_segment(source, node),
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        has_docstring=has_docstring,
                        docstring=docstring
                    )
                    self.components[func_id] = component

            elif isinstance(node, ast.Assign):
                # top-level assignments
                if hasattr(node, 'parent') and isinstance(node.parent, ast.Module):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assign_id = f"{module_path}.{target.id}"
                            component = CodeComponent(
                                id=assign_id,
                                node=node,
                                component_type="assignment",
                                file_path=file_path,
                                relative_path=relative_path,
                                source_code=self._get_source_segment(source, node),
                                start_line=node.lineno,
                                end_line=getattr(node, "end_lineno", node.lineno)
                            )
                            self.components[assign_id] = component

    def _resolve_dependencies(self):
        """
        Analyze each collected component's AST node to discover dependencies.
        """
        for component_id, component in list(self.components.items()):
            file_path = component.file_path

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()

                tree = ast.parse(source)
                add_parent_to_nodes(tree)

                # Collect imports for this file (with repo context)
                import_collector = ImportCollector(self._file_to_module_path(component.relative_path), self.modules)
                import_collector.visit(tree)

                # Identify the relevant AST node for this component
                component_node = None
                module_path = self._file_to_module_path(component.relative_path)

                if component.component_type == "function":
                    for node in ast.iter_child_nodes(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == component.id.split(".")[-1]:
                            component_node = node
                            break

                elif component.component_type == "assignment":
                    for node in ast.iter_child_nodes(tree):
                        if isinstance(node, ast.Assign):
                            if any(isinstance(t, ast.Name) and f"{module_path}.{t.id}" == component_id for t in node.targets):
                                component_node = node
                                break

                elif component.component_type == "class":
                    for node in ast.iter_child_nodes(tree):
                        if isinstance(node, ast.ClassDef) and node.name == component.id.split(".")[-1]:
                            component_node = node
                            break

                elif component.component_type == "method":
                    # Find method inside class
                    class_name, method_name = component.id.split(".")[-2:]
                    class_node = None
                    
                    for node in ast.iter_child_nodes(tree):
                        if isinstance(node, ast.ClassDef) and node.name == class_name:
                            class_node = node
                            for item in node.body:
                                if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) 
                                        and item.name == method_name):
                                    component_node = item
                                    break
                            break
                
                if component_node:
                    # Collect dependencies for this specific component
                    dependency_collector = DependencyCollector(
                        import_collector.imports,
                        import_collector.from_imports,
                        module_path,
                        self.modules
                    )
                    
                    # For functions and methods, collect variables defined in the function
                    if isinstance(component_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Add function parameters to local variables
                        for arg in component_node.args.args:
                            dependency_collector.local_variables.add(arg.arg)
                    
                    elif hasattr(component_node, "args"):
                        for arg in component_node.args.args:
                            dependency_collector.local_variables.add(arg.arg)
                            
                    dependency_collector.visit(component_node)
                    
                    # Add dependencies to the component
                    component.depends_on.update(dependency_collector.dependencies)
                    
                    # Filter out non-existent dependencies
                    component.depends_on = {
                        dep for dep in component.depends_on 
                        if dep in self.components or dep.split(".", 1)[0] in self.modules
                    }
                
            except (SyntaxError, UnicodeDecodeError) as e:
                logger.warning(f"Error analyzing dependencies in {file_path}: {e}")

    def _add_class_method_dependencies(self):
        """
        Make classes depend on their methods (except __init__).
        """
        class_methods: Dict[str, List[str]] = {}
        for component_id, component in self.components.items():
            if component.component_type == "method":
                parts = component_id.split(".")
                if len(parts) >= 2:
                    method_name = parts[-1]
                    class_id = ".".join(parts[:-1])
                    
                    if class_id not in class_methods:
                        class_methods[class_id] = []
                    
                    # Don't include __init__ methods as dependencies of the class
                    if method_name != "__init__":
                        class_methods[class_id].append(component_id)
        
        # Add method dependencies to their classes
        for class_id, method_ids in class_methods.items():
            if class_id in self.components:
                class_component = self.components[class_id]
                for method_id in method_ids:
                    class_component.depends_on.add(method_id)

    def _get_source_segment(self, source: str, node: ast.AST) -> str:
        try:
            if hasattr(ast, "get_source_segment"):
                segment = ast.get_source_segment(source, node)
                if segment is not None:
                    return segment
            lines = source.split("\n")
            start_line = node.lineno - 1
            end_line = getattr(node, "end_lineno", node.lineno) - 1
            return "\n".join(lines[start_line:end_line + 1])
        except Exception as e:
            logger.warning(f"Error getting source segment: {e}")
            return ""

    def _get_docstring(self, source: str, node: ast.AST) -> str:
        try:
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                for item in node.body:
                    if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
                        if isinstance(item.value.value, str):
                            return item.value.value
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
                        if isinstance(item.value.value, str):
                            return item.value.value
            return ""
        except Exception as e:
            logger.warning(f"Error getting docstring: {e}")
            return ""

    def save_dependency_graph(self, output_path: str):
        serializable_components = {comp_id: comp.to_dict() for comp_id, comp in self.components.items()}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_components, f, indent=2)
        logger.info(f"Saved dependency graph to {output_path}")

    def load_dependency_graph(self, input_path: str):
        with open(input_path, "r", encoding="utf-8") as f:
            serialized_components = json.load(f)
        self.components = {
            comp_id: CodeComponent.from_dict(comp_data)
            for comp_id, comp_data in serialized_components.items()
        }
        logger.info(f"Loaded {len(self.components)} components from {input_path}")
        return self.components
