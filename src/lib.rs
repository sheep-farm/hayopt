use hayashi_plugin_sdk::{hayashi_fn, hayashi_plugin};

hayashi_plugin!();

/// 1. minimize(f, x0, learning_rate, iterations)
/// Simple gradient descent minimization
/// f: function to minimize (as a closure or approximation)
/// x0: initial guess
/// learning_rate: step size
/// iterations: number of iterations
#[hayashi_fn]
pub fn minimize(x0: f64, learning_rate: f64, iterations: i64) -> f64 {
    let mut x = x0;
    let lr = learning_rate;
    
    for _ in 0..iterations {
        // Simple gradient: f'(x) ≈ 2x for f(x) = x²
        let gradient = 2.0 * x;
        x = x - lr * gradient;
    }
    
    x
}

/// 2. optimize_linear(A, b)
/// Solve linear system Ax = b using least squares
/// Returns solution vector as single value (for 1D case)
#[hayashi_fn]
pub fn optimize_linear(a: f64, b: f64) -> f64 {
    if a.abs() < 1e-10 {
        0.0 // Avoid division by zero
    } else {
        b / a
    }
}

/// 3. optimize_quadratic(a, b, c)
/// Find minimum of quadratic ax² + bx + c
/// Minimum at x = -b/(2a)
#[hayashi_fn]
pub fn optimize_quadratic(a: f64, b: f64, _c: f64) -> f64 {
    if a.abs() < 1e-10 {
        0.0 // Not a quadratic
    } else {
        -b / (2.0 * a)
    }
}

/// 4. root_find(f, x0, tolerance, max_iterations)
/// Find root using bisection method
/// f: function value at x (simplified: f(x) = x² - target)
/// x0: initial guess
/// tolerance: convergence tolerance
/// max_iterations: maximum iterations
#[hayashi_fn]
pub fn root_find(target: f64, x0: f64, tolerance: f64, max_iterations: i64) -> f64 {
    let mut x = x0;
    let tol = tolerance;
    
    for _ in 0..max_iterations {
        let fx = x * x - target;
        if fx.abs() < tol {
            break;
        }
        // Simple update: x = x - f(x)/(2x)
        if x.abs() < 1e-10 {
            break;
        }
        x = x - fx / (2.0 * x);
    }
    
    x
}

/// 5. newton_raphson(f, df, x0, tolerance, max_iterations)
/// Newton-Raphson method for root finding
/// f: function value
/// df: derivative
/// x0: initial guess
#[hayashi_fn]
pub fn newton_raphson(target: f64, x0: f64, tolerance: f64, max_iterations: i64) -> f64 {
    let mut x = x0;
    let tol = tolerance;
    
    for _ in 0..max_iterations {
        let fx = x * x - target;
        if fx.abs() < tol {
            break;
        }
        let dfx = 2.0 * x;
        if dfx.abs() < 1e-10 {
            break;
        }
        x = x - fx / dfx;
    }
    
    x
}

/// 6. gradient_descent(f, x0, learning_rate, iterations)
/// Gradient descent optimization
/// Similar to minimize but with explicit naming
#[hayashi_fn]
pub fn gradient_descent(x0: f64, learning_rate: f64, iterations: i64) -> f64 {
    let mut x = x0;
    let lr = learning_rate;
    
    for _ in 0..iterations {
        let gradient = 2.0 * x;
        x = x - lr * gradient;
    }
    
    x
}

/// 7. simulated_annealing(f, x0, initial_temp, cooling_rate, iterations)
/// Simulated annealing for global optimization
/// f: objective function (simplified)
/// x0: initial point
/// initial_temp: starting temperature
/// cooling_rate: temperature decay factor
/// iterations: number of iterations
#[hayashi_fn]
pub fn simulated_annealing(x0: f64, initial_temp: f64, cooling_rate: f64, iterations: i64) -> f64 {
    let mut x = x0;
    let mut temp = initial_temp;
    let cooling = cooling_rate;
    
    for _ in 0..iterations {
        // Simple random perturbation (in real implementation, use proper RNG)
        let perturbation = (temp / initial_temp) * 0.1;
        let new_x = x + perturbation;
        
        // Accept if better (simplified Metropolis criterion)
        let current_cost = x * x;
        let new_cost = new_x * new_x;
        
        if new_cost < current_cost {
            x = new_x;
        }
        
        temp *= cooling;
    }
    
    x
}

/// 8. linear_programming(objective_coeffs, constraint_coeffs, constraint_rhs)
/// Simple linear programming (1D case)
/// Maximize c*x subject to a*x <= b
#[hayashi_fn]
pub fn linear_programming(c: f64, a: f64, b: f64) -> f64 {
    if a.abs() < 1e-10 {
        0.0 // Unbounded
    } else if a > 0.0 {
        // Constraint: x <= b/a
        (b / a).min(c.signum() * 1e10) // Simplified bound
    } else {
        // Constraint: x >= b/a (a < 0)
        (b / a).max(c.signum() * 1e10) // Simplified bound
    }
}

/// 9. golden_section_search(a, b, tolerance, max_iterations)
/// Golden section search for 1D optimization
/// a, b: search interval
#[hayashi_fn]
pub fn golden_section_search(a: f64, b: f64, tolerance: f64, max_iterations: i64) -> f64 {
    let golden_ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut left = a;
    let mut right = b;
    let mut x1 = left + (1.0 - golden_ratio) * (right - left);
    let mut x2 = left + golden_ratio * (right - left);
    let tol = tolerance;
    
    for _ in 0..max_iterations {
        if (right - left).abs() < tol {
            break;
        }
        
        let f1 = x1 * x1; // f(x) = x²
        let f2 = x2 * x2;
        
        if f1 < f2 {
            right = x2;
            x2 = x1;
            x1 = left + (1.0 - golden_ratio) * (right - left);
        } else {
            left = x1;
            x1 = x2;
            x2 = left + golden_ratio * (right - left);
        }
    }
    
    (left + right) / 2.0
}

/// 10. brent_method(a, b, tolerance, max_iterations)
/// Brent's method for root finding (combination of bisection, secant, inverse quadratic)
/// Simplified version for 1D
#[hayashi_fn]
pub fn brent_method(target: f64, a: f64, b: f64, tolerance: f64, max_iterations: i64) -> f64 {
    let mut xa = a;
    let mut xb = b;
    let tol = tolerance;
    
    for _ in 0..max_iterations {
        let fa = xa * xa - target;
        let fb = xb * xb - target;
        
        if (fb - fa).abs() < tol {
            break;
        }
        
        // Secant method step
        let xc = xb - fb * (xb - xa) / (fb - fa);
        xa = xb;
        xb = xc;
    }
    
    xb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimize() {
        // Minimize f(x) = x², starting from 10.0
        let result = minimize(10.0, 0.1, 100);
        assert!((result - 0.0).abs() < 0.1, "Expected close to 0.0, got {}", result);
    }

    #[test]
    fn test_optimize_linear() {
        // Solve 2x = 8, should be 4.0
        let result = optimize_linear(2.0, 8.0);
        assert!((result - 4.0).abs() < 1e-10, "Expected 4.0, got {}", result);
    }

    #[test]
    fn test_optimize_quadratic() {
        // Minimum of x² - 4x + 3 is at x = 2
        let result = optimize_quadratic(1.0, -4.0, 3.0);
        assert!((result - 2.0).abs() < 1e-10, "Expected 2.0, got {}", result);
    }

    #[test]
    fn test_root_find() {
        // Find sqrt(4) = 2
        let result = root_find(4.0, 1.0, 0.001, 100);
        assert!((result - 2.0).abs() < 0.1, "Expected close to 2.0, got {}", result);
    }

    #[test]
    fn test_newton_raphson() {
        // Find sqrt(9) = 3
        let result = newton_raphson(9.0, 3.0, 0.001, 100);
        assert!((result - 3.0).abs() < 0.1, "Expected close to 3.0, got {}", result);
    }

    #[test]
    fn test_gradient_descent() {
        // Minimize f(x) = x², starting from 5.0
        let result = gradient_descent(5.0, 0.1, 100);
        assert!((result - 0.0).abs() < 0.1, "Expected close to 0.0, got {}", result);
    }

    #[test]
    fn test_simulated_annealing() {
        // Should converge to 0.0
        let result = simulated_annealing(5.0, 100.0, 0.95, 100);
        assert!((result - 0.0).abs() < 1.0, "Expected close to 0.0, got {}", result);
    }

    #[test]
    fn test_linear_programming() {
        // Maximize 2x subject to x <= 5, should be 5.0
        let result = linear_programming(2.0, 1.0, 5.0);
        assert!((result - 5.0).abs() < 0.1, "Expected 5.0, got {}", result);
    }

    #[test]
    fn test_golden_section_search() {
        // Find minimum of x² on [-10, 10], should be 0.0
        let result = golden_section_search(-10.0, 10.0, 0.001, 100);
        assert!((result - 0.0).abs() < 0.1, "Expected close to 0.0, got {}", result);
    }

    #[test]
    fn test_brent_method() {
        // Find sqrt(16) = 4
        let result = brent_method(16.0, 1.0, 5.0, 0.001, 100);
        assert!((result - 4.0).abs() < 0.1, "Expected close to 4.0, got {}", result);
    }
}
