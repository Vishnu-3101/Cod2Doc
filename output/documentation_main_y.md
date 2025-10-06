
        `
        main.y
        `

        This is the entry point of the code. The detailed explanation is provided below.
        
----------------------
### `y = SimpleFunction(x1,x2)`
```python
y = SimpleFunction(x1,x2)
```

This line calls the function `SimpleFunction` with arguments `x1` and `x2`, and assigns the returned value to the variable `y`.

-   `SimpleFunction`: This is a function call. The details of `SimpleFunction`'s implementation, including what it does with `x1` and `x2`, are documented in detail in the section describing `main.SimpleFunction`.
-   `x1`: This is the first argument passed to the `SimpleFunction`. More information about `x1` can be found in the section describing `main.x1`.
-   `x2`: This is the second argument passed to the `SimpleFunction`. More information about `x2` can be found in the section describing `main.x2`.
-   `y`: This variable stores the result returned by the `SimpleFunction` after its execution. The type and value of `y` depend on the implementation of the `SimpleFunction` and the values of `x1` and `x2`.
----------------------
### SimpleFunction
```python
def SimpleFunction(x1,x2):
    # Replace this function with your own function declaration.
    func =  (x1*x2)*(x1+x2)
    return func
```

The `SimpleFunction` takes two input arguments `x1` and `x2` and calculates a result based on them. Let's break down the code step by step:

-   `def SimpleFunction(x1, x2):`: This line defines a function named `SimpleFunction` that accepts two arguments, `x1` and `x2`. These arguments are expected to be numerical values.
-   `func = (x1*x2)*(x1+x2)`: This line performs the core calculation of the function. It calculates the product of `x1` and `x2`, then calculates the sum of `x1` and `x2`, and finally multiplies these two results together. The result of this calculation is stored in a variable named `func`.
-   `return func`: This line returns the calculated value `func` as the output of the function.
----------------------
### `x1 = Tensor(2)`
```python
x1 = Tensor(2)
```

This line initializes a `Tensor` object named `x1` with the value 2.

-   `x1`: This is the variable name to which the `Tensor` object will be assigned.
-   `Tensor(2)`: This part creates a new `Tensor` object. The `Tensor` class likely comes from a library like `AutoDiff` (as indicated by `dependent_comps`). The argument `2` is the initial value that the `Tensor` will hold. The detailed explanation of `AutoDiff.Tensor` will be explained further.

### `AutoDiff.Tensor`
```python
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
```

The `Tensor` class is designed to hold numerical data and track operations performed on it, which is essential for automatic differentiation. Let's break down the `__init__` method:

-   `def __init__(self, data, requires_grad=False):`: This is the constructor of the `Tensor` class. It initializes a new `Tensor` object.
    -   `self`: Refers to the instance of the `Tensor` class being created.
    -   `data`: This is the numerical value that the `Tensor` will hold. It could be a scalar (single number), vector, matrix, or even a higher-dimensional array.
    -   `requires_grad`: This is a boolean flag indicating whether gradients should be tracked for this `Tensor`. If `True`, the `Tensor` will record the operations performed on it so that gradients can be computed later using backpropagation. It defaults to `False` if not specified.
-   `self.data = data`: This line assigns the input `data` to the `data` attribute of the `Tensor` object. This is where the actual numerical value is stored.
-   `self.requires_grad = requires_grad`: This line assigns the input `requires_grad` to the `requires_grad` attribute of the `Tensor` object.
-   `self.grad = None`: This line initializes the `grad` attribute to `None`. The `grad` attribute will store the gradient of this `Tensor` with respect to some loss function. It's initially set to `None` because the gradient hasn't been computed yet.
-   `self._grad_fn = None`: This line initializes the `_grad_fn` attribute to `None`. The `_grad_fn` attribute will store a function that can compute the gradient of the operations that produced this `Tensor`. It's used during backpropagation to chain together the gradients of different operations. It's initially set to `None` because this `Tensor` was directly initialized with a value (in this case 2) and wasn't the result of any operation.
----------------------
### `class Tensor()`
```python
class Tensor():
    '''
    Converts each Tensor into a class of type Tensor.
    Operator overloading happens for each operator.
    exp: 
        x1+x2 => x1 is self
                 x2 is other
    After every operation, a new result node is created with operand nodes set to prev. This helps during backpropagation from destination node to source node to calculate gradients.
    '''
    def __init__(self,data, _prev=(),_op=''):
        '''
        Args:
            data: Holds the data of the node
            grad: The gradient Tensor
            _backward: Function to calculate gradients of prev nodes
                    => This is called in the one of the prev node operation which is responsible to create the current node.
            _prev: Keeps track of the previous nodes that generated the current node
            _op: Holds the operation performed on prev nodes to generate current node
        '''
        self.data = data
        self.grad = 0
        self._backward = lambda : None
        self._prev = set(_prev)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self,other),'+')
        def _backward():
            '''This function is executed only when _backward is called explictly or when node._backward() is called. When node._add__ is called, this function don't get executed. It will only be initialized to node._backward = 'function address'
            '''
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self,other),'*')
        def _backward():
            self.grad += out.grad*other.data
            other.grad += out.grad*self.data
        out._backward = _backward
        return out
    

    '''The __radd__ and __rmul__ represents reverse addition and multiplication
    Ex: 5+x1, 5 of type int cannot be added with a Tensor class.
        So in such cases __radd__ is called which converts 5+x1 to x1+5.
    '''
    def __radd__(self,other):
        return self+other
    
    def __rmul__(self,other):
        return self+other
    
    def backward(self):
        '''Perfrom topological sort using DFS.
        For every directed edge u-v, vertex u comes before v in the ordering.
        '''
        visited = set()
        topo = []

        def build_grad(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_grad(child)
                topo.append(node)

        '''The gradient of output node is set to 1. 
        Since backward is called only once, it is declared here..
        '''
        self.grad = 1             
        build_grad(self)
        for node in reversed(topo):
            node._backward()
```

The `Tensor` class is a fundamental building block for automatic differentiation. It encapsulates numerical data and enables the tracking of operations performed on that data, which is crucial for calculating gradients during backpropagation. The class overloads standard operators like addition and multiplication to create a computational graph.

The class consists of the following methods:

### `Tensor.__init__(self, data, _prev=(), _op='')`
```python
    def __init__(self,data, _prev=(),_op=''):
        '''
        Args:
            data: Holds the data of the node
            grad: The gradient Tensor
            _backward: Function to calculate gradients of prev nodes
                    => This is called in the one of the prev node operation which is responsible to create the current node.
            _prev: Keeps track of the previous nodes that generated the current node
            _op: Holds the operation performed on prev nodes to generate current node
        '''
        self.data = data
        self.grad = 0
        self._backward = lambda : None
        self._prev = set(_prev)
        self._op = _op
```
The `__init__` method initializes a new `Tensor` object.

-   `self`: Refers to the instance of the `Tensor` class being created.
-   `data`: This is the numerical value that the `Tensor` will hold. It could be a scalar (single number), vector, matrix, or even a higher-dimensional array. This is similar to the `data` attribute in the previously documented `AutoDiff.Tensor`, but without the `requires_grad` parameter.
-   `_prev`: This argument defaults to an empty tuple `()`. It is designed to store the "parent" `Tensor` objects that were used to create the current `Tensor`. This is essential for building the computational graph, which is used during backpropagation. The detailed explanation of `_prev` will be explained further.
-   `_op`: This argument defaults to an empty string `''`. It stores the operation that was performed to create the current `Tensor`. For example, if the current `Tensor` is the result of adding two other `Tensor` objects, `_op` would be set to `'+'`. This information is used during backpropagation to determine how to calculate the gradients. The detailed explanation of `_op` will be explained further.
-   `self.data = data`: This line assigns the input `data` to the `data` attribute of the `Tensor` object. This is where the actual numerical value is stored.
-   `self.grad = 0`: This line initializes the `grad` attribute to `0`. The `grad` attribute will store the gradient of this `Tensor` with respect to some loss function. It's initially set to `0` because the gradient hasn't been computed yet.
-   `self._backward = lambda : None`:  This line initializes the `_backward` attribute to a lambda function that does nothing. The `_backward` attribute will store a function that calculates the gradient of the current `Tensor` with respect to its inputs (i.e., the `_prev` `Tensor` objects). This function is used during backpropagation. It's initialized to a no-op lambda function because the gradient calculation depends on the operation that was performed to create the `Tensor`. The detailed explanation of `_backward` will be explained further.
-   `self._prev = set(_prev)`: This line converts the `_prev` tuple to a set and assigns it to the `_prev` attribute. Storing the previous nodes as a set ensures that there are no duplicate entries, which can be important for efficiency during backpropagation. The detailed explanation of `_prev` will be explained further.
-   `self._op = _op`: This line assigns the input `_op` to the `_op` attribute of the `Tensor` object. The detailed explanation of `_op` will be explained further.

### `Tensor.__repr__(self)`
```python
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```
The `__repr__` method defines how a `Tensor` object is represented as a string.

-   `self`: Refers to the instance of the `Tensor` class.
-   `return f"Value(data={self.data}, grad={self.grad})"`: This line returns a formatted string that displays the `data` and `grad` attributes of the `Tensor` object. This is useful for debugging and inspecting the values of `Tensor` objects.

### `Tensor.__add__(self, other)`
```python
    def __add__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self,other),'+')
        def _backward():
            '''This function is executed only when _backward is called explictly or when node._backward() is called. When node._add__ is called, this function don't get executed. It will only be initialized to node._backward = 'function address'
            '''
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
```
The `__add__` method overloads the addition operator (`+`) for `Tensor` objects.

-   `self`: Refers to the `Tensor` object on the left-hand side of the `+` operator (e.g., `x1` in `x1 + x2`).
-   `other`: Refers to the object on the right-hand side of the `+` operator (e.g., `x2` in `x1 + x2`).
-   `other = other if isinstance(other,Tensor) else Tensor(other)`: This line checks if `other` is a `Tensor` object. If not, it converts `other` to a `Tensor` object using the `Tensor` constructor. This allows you to add a `Tensor` with a scalar value (e.g., `x1 + 5`).
-   `out = Tensor(self.data + other.data, (self,other),'+')`: This line creates a new `Tensor` object named `out` to store the result of the addition.
    -   `self.data + other.data`: This calculates the sum of the `data` attributes of the two `Tensor` objects.
    -   `(self, other)`: This creates a tuple containing the two input `Tensor` objects (`self` and `other`). This tuple is assigned to the `_prev` attribute of the `out` `Tensor`, which is used to track the dependencies between `Tensor` objects for backpropagation. The detailed explanation of `_prev` will be explained further.
    -   `'+'`: This indicates that the operation performed was addition. This string is assigned to the `_op` attribute of the `out` `Tensor`. The detailed explanation of `_op` will be explained further.
-   The `_backward` function calculates the gradients of the inputs (`self` and `other`) with respect to the output (`out`).
    -   `self.grad += out.grad`: This line adds the gradient of the output (`out.grad`) to the gradient of the first input (`self.grad`).
    -   `other.grad += out.grad`: This line adds the gradient of the output (`out.grad`) to the gradient of the second input (`other.grad`).
-   `out._backward = _backward`: This line assigns the `_backward` function to the `_backward` attribute of the `out` `Tensor`. This allows the `_backward` function to be called during backpropagation to calculate the gradients of the inputs. The detailed explanation of `_backward` will be explained further.
-   `return out`: This line returns the new `Tensor` object `out` containing the result of the addition.

### `Tensor.__mul__(self, other)`
```python
    def __mul__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self,other),'*')
        def _backward():
            self.grad += out.grad*other.data
            other.grad += out.grad*self.data
        out._backward = _backward
        return out
```
The `__mul__` method overloads the multiplication operator (`*`) for `Tensor` objects.

-   `self`: Refers to the `Tensor` object on the left-hand side of the `*` operator (e.g., `x1` in `x1 * x2`).
-   `other`: Refers to the object on the right-hand side of the `*` operator (e.g., `x2` in `x1 * x2`).
-   `other = other if isinstance(other,Tensor) else Tensor(other)`: This line checks if `other` is a `Tensor` object. If not, it converts `other` to a `Tensor` object using the `Tensor` constructor. This allows you to multiply a `Tensor` with a scalar value (e.g., `x1 * 5`).
-   `out = Tensor(self.data * other.data, (self,other),'*')`: This line creates a new `Tensor` object named `out` to store the result of the multiplication.
    -   `self.data * other.data`: This calculates the product of the `data` attributes of the two `Tensor` objects.
    -   `(self, other)`: This creates a tuple containing the two input `Tensor` objects (`self` and `other`). This tuple is assigned to the `_prev` attribute of the `out` `Tensor`, which is used to track the dependencies between `Tensor` objects for backpropagation.
    -   `'*'`: This indicates that the operation performed was multiplication. This string is assigned to the `_op` attribute of the `out` `Tensor`.
-   The `_backward` function calculates the gradients of the inputs (`self` and `other`) with respect to the output (`out`).
    -   `self.grad += out.grad*other.data`: This line adds the gradient of the output (`out.grad`) multiplied by the data of the other tensor (`other.data`) to the gradient of the first input (`self.grad`).
    -   `other.grad += out.grad*self.data`: This line adds the gradient of the output (`out.grad`) multiplied by the data of the self tensor (`self.data`) to the gradient of the second input (`other.grad`).
-   `out._backward = _backward`: This line assigns the `_backward` function to the `_backward` attribute of the `out` `Tensor`. This allows the `_backward` function to be called during backpropagation to calculate the gradients of the inputs.
-   `return out`: This line returns the new `Tensor` object `out` containing the result of the multiplication.

### `Tensor.__radd__(self, other)`
```python
    def __radd__(self,other):
        return self+other
```

The `__radd__` method implements reverse addition. This is called when a `Tensor` object is on the right side of the `+` operator and the left side is not a `Tensor`.

-   `self`: Refers to the `Tensor` object.
-   `other`: Refers to the object on the left-hand side of the `+` operator (which is not a `Tensor`).
-   `return self + other`: This line simply calls the `__add__` method with the arguments swapped. This ensures that the addition operation is performed correctly, regardless of the order of the operands.

### `Tensor.__rmul__(self, other)`
```python
    def __rmul__(self,other):
        return self+other
```

The `__rmul__` method implements reverse multiplication. This is called when a `Tensor` object is on the right side of the `*` operator and the left side is not a `Tensor`.

-   `self`: Refers to the `Tensor` object.
-   `other`: Refers to the object on the left-hand side of the `*` operator (which is not a `Tensor`).
-   `return self + other`: This line simply calls the `__add__` method with the arguments swapped. This ensures that the addition operation is performed correctly, regardless of the order of the operands.

### `Tensor.backward(self)`
```python
    def backward(self):
        '''Perfrom topological sort using DFS.
        For every directed edge u-v, vertex u comes before v in the ordering.
        '''
        visited = set()
        topo = []

        def build_grad(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_grad(child)
                topo.append(node)

        '''The gradient of output node is set to 1. 
        Since backward is called only once, it is declared here..
        '''
        self.grad = 1             
        build_grad(self)
        for node in reversed(topo):
            node._backward()
```

The `backward` method performs backpropagation to calculate the gradients of all `Tensor` objects in the computational graph with respect to the output `Tensor`.

-   `self`: Refers to the output `Tensor` object from which backpropagation is initiated.
-   `visited = set()`: Initializes an empty set called `visited`. This set is used to keep track of the nodes that have already been visited during the topological sort.
-   `topo = []`: Initializes an empty list called `topo`. This list will store the nodes in topological order.
-   The `build_grad` function performs a depth-first search (DFS) to build the topological order of the computational graph.
    -   `def build_grad(node):`: Defines a function called `build_grad` that takes a `node` as input.
    -   `if node not in visited:`: Checks if the current `node` has already been visited.
    -   `visited.add(node)`: Adds the current `node` to the `visited` set.
    -   `for child in node._prev:`: Iterates over the children of the current `node`. The detailed explanation of `_prev` will be explained further.
        -   `build_grad(child)`: Recursively calls the `build_grad` function on each child node. The detailed explanation of `child` will be explained further.
    -   `topo.append(node)`: Appends the current `node` to the `topo` list after all of its children have been visited.
-   `self.grad = 1`: This line sets the gradient of the output `Tensor` to 1.0. This is because the gradient of a `Tensor` with respect to itself is always 1.
-   `build_grad(self)`: This line calls the `build_grad` function on the output `Tensor` to start the topological sort.
-   `for node in reversed(topo):`: This line iterates over the nodes in the `topo` list in reverse order. This ensures that the gradients are calculated in the correct order during backpropagation.
    -   `node._backward()`: This line calls the `_backward` function of the current `node`. This function calculates the gradients of the inputs to the current `node` with respect to the current `node`. The detailed explanation of `_backward` will be explained further.
----------------------
### `Tensor.__add__(self, other)`
```python
    def __add__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self,other),'+')
        def _backward():
            '''This function is executed only when _backward is called explictly or when node._backward() is called. When node._add__ is called, this function don't get executed. It will only be initialized to node._backward = 'function address'
            '''
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
```
The `__add__` method overloads the addition operator (`+`) for `Tensor` objects.

-   `self`: Refers to the `Tensor` object on the left-hand side of the `+` operator (e.g., `x1` in `x1 + x2`).
-   `other`: Refers to the object on the right-hand side of the `+` operator (e.g., `x2` in `x1 + x2`).
-   `other = other if isinstance(other,Tensor) else Tensor(other)`: This line checks if `other` is a `Tensor` object. If not, it converts `other` to a `Tensor` object using the `Tensor` constructor. This allows you to add a `Tensor` with a scalar value (e.g., `x1 + 5`). The detailed explanation of `AutoDiff.Tensor` will be explained further.
-   `out = Tensor(self.data + other.data, (self,other),'+')`: This line creates a new `Tensor` object named `out` to store the result of the addition.
    -   `self.data + other.data`: This calculates the sum of the `data` attributes of the two `Tensor` objects.
    -   `(self, other)`: This creates a tuple containing the two input `Tensor` objects (`self` and `other`). This tuple is assigned to the `_prev` attribute of the `out` `Tensor`, which is used to track the dependencies between `Tensor` objects for backpropagation.
    -   `'+'`: This indicates that the operation performed was addition. This string is assigned to the `_op` attribute of the `out` `Tensor`.
-   The `_backward` function calculates the gradients of the inputs (`self` and `other`) with respect to the output (`out`).
    -   `self.grad += out.grad`: This line adds the gradient of the output (`out.grad`) to the gradient of the first input (`self.grad`).
    -   `other.grad += out.grad`: This line adds the gradient of the output (`out.grad`) to the gradient of the second input (`other.grad`).
-   `out._backward = _backward`: This line assigns the `_backward` function to the `_backward` attribute of the `out` `Tensor`. This allows the `_backward` function to be called during backpropagation to calculate the gradients of the inputs. The detailed explanation of `AutoDiff._backward` will be explained further.
-   `return out`: This line returns the new `Tensor` object `out` containing the result of the addition.
----------------------
### `Tensor.__repr__(self)`
```python
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```
The `__repr__` method defines how a `Tensor` object is represented as a string.

-   `self`: Refers to the instance of the `Tensor` class.
-   `return f"Value(data={self.data}, grad={self.grad})"`: This line returns a formatted string that displays the `data` and `grad` attributes of the `Tensor` object. This is useful for debugging and inspecting the values of `Tensor` objects.
----------------------
### `Tensor.__radd__(self, other)`
```python
    def __radd__(self,other):
        return self+other
```

The `__radd__` method implements reverse addition for the `Tensor` class. This method is invoked when a `Tensor` object appears on the right-hand side of the `+` operator, while the left-hand side operand is not a `Tensor`. It ensures that addition is commutative, allowing operations like `5 + x` (where `x` is a `Tensor`) to be performed correctly.

-   `self`: Refers to the instance of the `Tensor` class.
-   `other`: Represents the object on the left-hand side of the `+` operator, which is not a `Tensor`.
-   `return self + other`: This line leverages the existing `__add__` method to perform the addition. By returning `self + other`, it effectively converts the operation to `other + self`, allowing the `__add__` method to handle the addition, including the necessary gradient calculations for backpropagation.
----------------------
### `Tensor.__rmul__(self, other)`
```python
    def __rmul__(self,other):
        return self+other
```

The `__rmul__` method implements reverse multiplication. This is called when a `Tensor` object is on the right side of the `*` operator and the left side is not a `Tensor`.

-   `self`: Refers to the `Tensor` object.
-   `other`: Refers to the object on the left-hand side of the `*` operator (which is not a `Tensor`).
-   `return self + other`: This line simply calls the `__add__` method with the arguments swapped. This ensures that the addition operation is performed correctly, regardless of the order of the operands.
----------------------
### `Tensor.__mul__(self, other)`
```python
    def __mul__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self,other),'*')
        def _backward():
            self.grad += out.grad*other.data
            other.grad += out.grad*self.data
        out._backward = _backward
        return out
```
The `__mul__` method overloads the multiplication operator (`*`) for `Tensor` objects.

-   `self`: Refers to the `Tensor` object on the left-hand side of the `*` operator (e.g., `x1` in `x1 * x2`).
-   `other`: Refers to the object on the right-hand side of the `*` operator (e.g., `x2` in `x1 * x2`).
-   `other = other if isinstance(other,Tensor) else Tensor(other)`: This line checks if `other` is a `Tensor` object. If not, it converts `other` to a `Tensor` object using the `Tensor` constructor. This allows you to multiply a `Tensor` with a scalar value (e.g., `x1 * 5`). The detailed explanation of `AutoDiff.Tensor` will be explained further.
-   `out = Tensor(self.data * other.data, (self,other),'*')`: This line creates a new `Tensor` object named `out` to store the result of the multiplication.
    -   `self.data * other.data`: This calculates the product of the `data` attributes of the two `Tensor` objects.
    -   `(self, other)`: This creates a tuple containing the two input `Tensor` objects (`self` and `other`). This tuple is assigned to the `_prev` attribute of the `out` `Tensor`, which is used to track the dependencies between `Tensor` objects for backpropagation.
    -   `'*'`: This indicates that the operation performed was multiplication. This string is assigned to the `_op` attribute of the `out` `Tensor`.
-   The `_backward` function calculates the gradients of the inputs (`self` and `other`) with respect to the output (`out`).
    -   `self.grad += out.grad*other.data`: This line adds the gradient of the output (`out.grad`) multiplied by the data of the other tensor (`other.data`) to the gradient of the first input (`self.grad`).
    -   `other.grad += out.grad*self.data`: This line adds the gradient of the output (`out.grad`) multiplied by the data of the self tensor (`self.data`) to the gradient of the second input (`other.grad`).
-   `out._backward = _backward`: This line assigns the `_backward` function to the `_backward` attribute of the `out` `Tensor`. This allows the `_backward` function to be called during backpropagation to calculate the gradients of the inputs. The detailed explanation of `AutoDiff._backward` will be explained further.
-   `return out`: This line returns the new `Tensor` object `out` containing the result of the multiplication.
----------------------
### `Tensor.backward(self)`
```python
    def backward(self):
        '''Perfrom topological sort using DFS.
        For every directed edge u-v, vertex u comes before v in the ordering.
        '''
        visited = set()
        topo = []

        def build_grad(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_grad(child)
                topo.append(node)

        '''The gradient of output node is set to 1. 
        Since backward is called only once, it is declared here..
        '''
        self.grad = 1             
        build_grad(self)
        for node in reversed(topo):
            node._backward()
```

The `backward` method performs backpropagation to calculate the gradients of all `Tensor` objects in the computational graph with respect to the output `Tensor`.

-   `self`: Refers to the output `Tensor` object from which backpropagation is initiated.
-   `visited = set()`: Initializes an empty set called `visited`. This set is used to keep track of the nodes that have already been visited during the topological sort.
-   `topo = []`: Initializes an empty list called `topo`. This list will store the nodes in topological order.
-   The `build_grad` function performs a depth-first search (DFS) to build the topological order of the computational graph.
    -   `def build_grad(node):`: Defines a function called `build_grad` that takes a `node` as input. The detailed explanation of `AutoDiff.node` will be explained further.
    -   `if node not in visited:`: Checks if the current `node` has already been visited.
    -   `visited.add(node)`: Adds the current `node` to the `visited` set.
    -   `for child in node._prev:`: Iterates over the children of the current `node`. The detailed explanation of `AutoDiff.child` and `_prev` will be explained further.
        -   `build_grad(child)`: Recursively calls the `build_grad` function on each child node.
    -   `topo.append(node)`: Appends the current `node` to the `topo` list after all of its children have been visited.
-   `self.grad = 1`: This line sets the gradient of the output `Tensor` to 1. This is because the gradient of a `Tensor` with respect to itself is always 1.
-   `build_grad(self)`: This line calls the `build_grad` function on the output `Tensor` to start the topological sort. The detailed explanation of `AutoDiff.build_grad` will be explained further.
-   `for node in reversed(topo):`: This line iterates over the nodes in the `topo` list in reverse order. This ensures that the gradients are calculated in the correct order during backpropagation.
    -   `node._backward()`: This line calls the `_backward` function of the current `node`. This function calculates the gradients of the inputs to the current `node` with respect to the current `node`. The detailed explanation of `AutoDiff._backward` will be explained further.

### `AutoDiff.build_grad(node)`
The `build_grad` function is a recursive function that performs a depth-first search (DFS) on the computational graph to determine the correct order for backpropagation. The goal is to ensure that the gradients are computed in the reverse order of the operations performed during the forward pass.

-   `node`: Represents the current node being visited in the computational graph. Each node corresponds to a `Tensor` object.
-   `if node not in visited:`: This condition checks whether the current node has been visited before. The `visited` set keeps track of all nodes that have already been processed. This check is crucial to prevent infinite loops in the case of cyclic graphs and to ensure that each node is processed only once.
-   `visited.add(node)`: If the node hasn't been visited, it's added to the `visited` set to mark it as processed.
-   `for child in node._prev:`: This loop iterates over the children of the current node. The `_prev` attribute of a `Tensor` object stores the `Tensor` objects that were used to create it. In other words, the children are the inputs to the operation that produced the current node.
-   `build_grad(child)`: This line recursively calls the `build_grad` function on each child node. This ensures that all descendants of the current node are visited before the current node itself.
-   `topo.append(node)`: After all the children of the current node have been visited, the current node is appended to the `topo` list. This list maintains the topological order of the nodes in the computational graph.

### `AutoDiff.node` and `AutoDiff.child`
In the context of the `build_grad` function and the `Tensor` class, both "node" and "child" refer to `Tensor` objects within the computational graph.

-   `node`: Generally refers to the current `Tensor` object being processed by the `build_grad` function or within the broader context of backpropagation. It represents a point in the computational graph where data is stored and gradients are calculated.
-   `child`: Specifically refers to the `Tensor` objects that were used as inputs to create another `Tensor` object (the "parent" or "node"). The `child` nodes are stored in the `_prev` attribute of the parent `Tensor`. During backpropagation, the gradients are propagated from the `node` to its `child` nodes.

### `AutoDiff._prev`
The `_prev` attribute of a `Tensor` object is a set that stores the `Tensor` objects that were used as inputs to create the current `Tensor`. It is a crucial component of the computational graph, as it allows the `backward` method to trace the dependencies between `Tensor` objects and calculate gradients correctly.

For example, if `z = x + y`, then the `_prev` attribute of `z` would be a set containing `x` and `y`: `z._prev = {x, y}`.

During backpropagation, the `backward` method iterates over the `_prev` set of each `Tensor` object to propagate the gradients to its inputs.

### `AutoDiff._backward`
The `_backward` attribute of a `Tensor` object is a function that calculates the gradients of the inputs to the `Tensor` with respect to the `Tensor` itself. This function is defined during the forward pass when the `Tensor` is created as the result of an operation (e.g., addition, multiplication).

For example, consider the addition operation `z = x + y`. The `_backward` function for `z` would calculate `dz/dx` and `dz/dy`, which are both 1 in this case. The `_backward` function would then add these gradients to the `grad` attributes of `x` and `y`, respectively.
----------------------
### `x2 = Tensor(3)`
```python
x2 = Tensor(3)
```

This line initializes a `Tensor` object named `x2` with the value 3.

-   `x2`: This is the variable name to which the `Tensor` object will be assigned.
-   `Tensor(3)`: This part creates a new `Tensor` object. The `Tensor` class likely comes from a library like `AutoDiff` (as indicated by `dependent_comps`). The argument `3` is the initial value that the `Tensor` will hold. The detailed explanation of `AutoDiff.Tensor` is documented previously.