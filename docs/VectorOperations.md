# Vector Operations Documentation

## Vector Initialization

### Basic Initialization Functions

| Function | Description | Mathematical Expression | Example |
|----------|-------------|------------------------|---------|
| `ca.zeros()` | Initialize with zeros | $\mathbf{v}_i = 0$ | `ca.vector(5, ca.zeros())` |
| `ca.ones()` | Initialize with ones | $\mathbf{v}_i = 1$ | `ca.vector(5, ca.ones())` |
| `ca.constant(c)` | Initialize with constant | $\mathbf{v}_i = c$ | `ca.vector(5, ca.constant(3.14))` |

### Random Initialization Functions

| Function | Description | Mathematical Expression | Example |
|----------|-------------|------------------------|---------|
| `ca.random(seed?)` | Uniform [0,1] | $\mathbf{v}_i \sim \mathcal{U}(0,1)$ | `ca.vector(5, ca.random(42))` |
| `ca.uniform(min, max, seed?)` | Uniform [min,max] | $\mathbf{v}_i \sim \mathcal{U}(a,b)$ | `ca.vector(5, ca.uniform(-1, 1))` |
| `ca.normal(μ, σ, seed?)` | Normal distribution | $\mathbf{v}_i \sim \mathcal{N}(\mu,\sigma^2)$ | `ca.vector(5, ca.normal(0, 1))` |
| `ca.box_muller(μ, σ, seed?)` | Box-Muller transform | $\mathbf{v}_i \sim \mathcal{N}(\mu,\sigma^2)$ | `ca.vector(5, ca.box_muller(0, 1))` |

### Mathematical Initialization Functions

| Function | Description | Mathematical Expression | Example |
|----------|-------------|------------------------|---------|
| `ca.sequence(start, step)` | Arithmetic sequence | $\mathbf{v}_i = a + i \cdot d$ | `ca.vector(5, ca.sequence(0, 2))` |
| `ca.arange(start, stop, step)` | Range sequence | $\mathbf{v}_i = a + i \cdot d$ | `ca.vector(5, ca.arange(0, 10, 2))` |
| `ca.sine(freq, phase)` | Sine wave | $\mathbf{v}_i = \sin(2\pi f \cdot i + \phi)$ | `ca.vector(5, ca.sine(1.0, 0))` |

## Vector Operations

### Element-wise Operations

| Operation | Function | Mathematical Expression | Example |
|-----------|----------|------------------------|---------|
| **Multiplication** | `ca.vecmul(a, b)` | $\mathbf{c}_i = \mathbf{a}_i \cdot \mathbf{b}_i$ | `c = ca.vecmul(a, b)` |
| **Addition** | `ca.vecadd(a, b)` | $\mathbf{c}_i = \mathbf{a}_i + \mathbf{b}_i$ | `c = ca.vecadd(a, b)` |
| **Subtraction** | `ca.vecsub(a, b)` | $\mathbf{c}_i = \mathbf{a}_i - \mathbf{b}_i$ | `c = ca.vecsub(a, b)` |

### Scalar Operations

| Operation | Function | Mathematical Expression | Example |
|-----------|----------|------------------------|---------|
| **Scalar Multiplication** | `ca.vecmul_scalar(a, s)` | $\mathbf{c}_i = \mathbf{a}_i \cdot s$ | `c = ca.vecmul_scalar(a, 2.5)` |
| **Scalar Addition** | `ca.vecadd_scalar(a, s)` | $\mathbf{c}_i = \mathbf{a}_i + s$ | `c = ca.vecadd_scalar(a, 1.0)` |

### Operator Overloading

| Operation | Operator | Mathematical Expression | Example |
|-----------|----------|------------------------|---------|
| **Vector × Vector** | `a * b` | $\mathbf{c}_i = \mathbf{a}_i \cdot \mathbf{b}_i$ | `c = a * b` |
| **Vector + Vector** | `a + b` | $\mathbf{c}_i = \mathbf{a}_i + \mathbf{b}_i$ | `c = a + b` |
| **Vector - Vector** | `a - b` | $\mathbf{c}_i = \mathbf{a}_i - \mathbf{b}_i$ | `c = a - b` |
| **Vector × Scalar** | `a * s` | $\mathbf{c}_i = \mathbf{a}_i \cdot s$ | `c = a * 2.5` |
| **Vector + Scalar** | `a + s` | $\mathbf{c}_i = \mathbf{a}_i + s$ | `c = a + 1.0` |
| **Scalar × Vector** | `s * a` | $\mathbf{c}_i = s \cdot \mathbf{a}_i$ | `c = 2.5 * a` |

## Usage Examples

### Basic Initialization
```python
import cato as ca

# Create vectors with different initializations
zeros = ca.vector(5, ca.zeros())           # [0, 0, 0, 0, 0]
ones = ca.vector(5, ca.ones())             # [1, 1, 1, 1, 1]
constant = ca.vector(5, ca.constant(3.14)) # [3.14, 3.14, 3.14, 3.14, 3.14]
random = ca.vector(5, ca.random(42))       # Random values [0,1]
normal = ca.vector(5, ca.normal(0, 1))     # Normal distribution N(0,1)
```

### Mathematical Sequences
```python
# Arithmetic sequences
seq = ca.vector(5, ca.sequence(0, 2))      # [0, 2, 4, 6, 8]
arange = ca.vector(5, ca.arange(0, 10, 2)) # [0, 2, 4, 6, 8]

# Sine wave
sine = ca.vector(5, ca.sine(1.0, 0))       # [0, 0.95, 0.59, -0.59, -0.95]
```

### Vector Operations
```python
# Create test vectors
a = ca.vector(3, ca.sequence(1, 1))        # [1, 2, 3]
b = ca.vector(3, ca.sequence(2, 1))        # [2, 3, 4]

# Element-wise operations
mul = ca.vecmul(a, b)                      # [2, 6, 12]
add = ca.vecadd(a, b)                      # [3, 5, 7]
sub = ca.vecsub(a, b)                      # [-1, -1, -1]

# Scalar operations
scalar_mul = ca.vecmul_scalar(a, 2.0)      # [2, 4, 6]
scalar_add = ca.vecadd_scalar(a, 1.0)      # [2, 3, 4]

# Using operators
mul_op = a * b                             # [2, 6, 12]
add_op = a + b                             # [3, 5, 7]
scalar_op = a * 2.0                        # [2, 4, 6]
```

## Performance Notes

- **GPU Acceleration**: All operations are automatically optimized for CUDA GPUs
- **Memory Management**: Automatic CPU ↔ GPU memory transfers
- **Kernel Optimization**: Launch parameters are automatically optimized
- **Type Safety**: All operations use `double` precision by default

## Requirements

- Vectors must have the same size for element-wise operations
- CUDA-compatible GPU required for optimal performance
- All operations return new vectors (no in-place modification)
