# fitResults

All optimizers return a struct fitResults.

- **value** fit: the final fit value (regularized fit)
- **value** fits: a vector with all fits at the outer iteration
- **value** convergence: was the outer breaking condition met?
- **value** parameterValues: final parameter values
- **value** Hessian: final Hessian approximation (optional)
