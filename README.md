# hayopt

Optimization plugin for Hayashi.

## Installation

```bash
hay install sheep-farm/hayopt
```

## Usage

```hayashi
import("sheep-farm/hayopt", as=opt)

// Simple gradient descent minimization
let result = opt::minimize(10.0, 0.1, 100)
print(result)

// Find minimum of quadratic ax² + bx + c
let min_x = opt::optimize_quadratic(1.0, -4.0, 3.0)
print(min_x)

// Solve linear system Ax = b
let solution = opt::optimize_linear(2.0, 8.0)
print(solution)

// Root finding using Newton-Raphson
let root = opt::newton_raphson(4.0, 1.0, 0.001, 100)
print(root)

// Golden section search
let optimum = opt::golden_section_search(-10.0, 10.0, 0.001, 100)
print(optimum)
```

## Functions

### Basic Optimization
- `minimize(x0, learning_rate, iterations)` - Gradient descent minimization
- `gradient_descent(x0, learning_rate, iterations)` - Explicit gradient descent

### Analytical Solutions
- `optimize_linear(a, b)` - Solve linear system Ax = b
- `optimize_quadratic(a, b, c)` - Find minimum of quadratic

### Root Finding
- `root_find(target, x0, tolerance, max_iterations)` - Bisection method
- `newton_raphson(target, x0, tolerance, max_iterations)` - Newton-Raphson method
- `brent_method(target, a, b, tolerance, max_iterations)` - Brent's method

### Global Optimization
- `simulated_annealing(x0, initial_temp, cooling_rate, iterations)` - Simulated annealing
- `golden_section_search(a, b, tolerance, max_iterations)` - Golden section search

### Constrained Optimization
- `linear_programming(c, a, b)` - Simple linear programming (1D)

## Development

```bash
cargo build --release
cp target/release/libhayopt.so ~/.hay/packages/sheep-farm/hayopt.so
```

## Dependencies

- **faer**: High-performance linear algebra library (no BLAS dependency)
- **hayashi-plugin-sdk**: Hayashi plugin SDK
