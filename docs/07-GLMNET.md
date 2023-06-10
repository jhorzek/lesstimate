# GLMNET 


The implementation of GLMNET follows that outlined in

1. Friedman, J., Hastie, T., & Tibshirani, R. (2010).
Regularization Paths for Generalized Linear Models via Coordinate Descent.
Journal of Statistical Software, 33(1), 1–20. https://doi.org/10.18637/jss.v033.i01
2. Yuan, G.-X., Chang, K.-W., Hsieh, C.-J., & Lin, C.-J. (2010).
A Comparison of Optimization Methods and Software for Large-scale
L1-regularized Linear Classification. Journal of Machine Learning Research, 11, 3183–3234.
3. Yuan, G.-X., Ho, C.-H., & Lin, C.-J. (2012).
An improved GLMNET for l1-regularized logistic regression.
The Journal of Machine Learning Research, 13, 1999–2030. https://doi.org/10.1145/2020408.2020421

## fitGlmnet

We provide two optimizer interfaces: One uses a combination of arma::rowvec and lessSEM::stringVector for starting
values and parameter labels respectively. This interface is consistent with the fit and gradient function of the
`lessSEM::model`-class. Alternatively, a numericVector can be passed to the optimizers. This design is rooted in
the use of Rcpp::NumericVectors that combine values and labels similar to an R vector. Thus, interfacing to this
second function call can be easier when coming from R.

### Version 1

- **param** userModel: your model. Must inherit from lessSEM::model!
- **param** startingValues: numericVector with initial starting values. This
vector can have names.
- **param** penalty: vector with strings indicating the penalty for each parameter.
Currently supported are "none", "cappedL1", "lasso", "lsp", "mcp", and "scad".
(e.g., {"none", "scad", "scad", "lasso", "none"}). If only one value is provided,
the same penalty will be applied to every parameter!
- **param** lambda: lambda tuning parameter values. One lambda value for each parameter.
If only one value is provided, this value will be applied to each parameter.
Important: The the function will _not_ loop over these values but assume that you
may want to provide different levels of regularization for each parameter!
- **param** theta: theta tuning parameter values. One theta value for each parameter
If only one value is provided, this value will be applied to each parameter.
Not all penalties use theta.
Important: The the function will _not_ loop over these values but assume that you
may want to provide different levels of regularization for each parameter!
- **param** initialHessian: matrix with initial Hessian values.
- **param** controlOptimizer: option to change the optimizer settings
- **param** verbose: should additional information be printed? If set > 0, additional
information will be provided. Highly recommended for initial runs. Note that
the optimizer itself has a separate verbose argument that can be used to print
information on each iteration. This can be set with the controlOptimizer - argument.
- **return** fitResults

### Version 2

- **param** userModel: your model. Must inherit from lessSEM::model!
- **param** startingValues: an arma::rowvec numeric vector with starting values
- **param** parameterLabels: a lessSEM::stringVector with labels for parameters
- **param** penalty: vector with strings indicating the penalty for each parameter.
Currently supported are "none", "cappedL1", "lasso", "lsp", "mcp", and "scad".
(e.g., {"none", "scad", "scad", "lasso", "none"}). If only one value is provided,
the same penalty will be applied to every parameter!
- **param** lambda: lambda tuning parameter values. One lambda value for each parameter.
If only one value is provided, this value will be applied to each parameter.
Important: The the function will _not_ loop over these values but assume that you
may want to provide different levels of regularization for each parameter!
- **param** theta: theta tuning parameter values. One theta value for each parameter
If only one value is provided, this value will be applied to each parameter.
Not all penalties use theta.
Important: The the function will _not_ loop over these values but assume that you
may want to provide different levels of regularization for each parameter!
- **param** initialHessian: matrix with initial Hessian values.
- **param** controlOptimizer: option to change the optimizer settings
- **param** verbose: should additional information be printed? If set > 0, additional
information will be provided. Highly recommended for initial runs. Note that
the optimizer itself has a separate verbose argument that can be used to print
information on each iteration. This can be set with the controlOptimizer - argument.
- **return** fitResults

## glmnet

Optimize a model using the glmnet procedure.

### Version 1


- **T-param** nonsmoothPenalty: class of type nonsmoothPenalty (e.g., lasso, scad, lsp)
- **T-param** smoothPenalty: class of type smooth penalty (e.g., ridge)
- **T-param** tuning: tuning parameters used by both, the nonsmootPenalty and the smoothPenalty
- **param** model_: the model object derived from the model class in model.h
- **param** startingValuesRcpp: an Rcpp numeric vector with starting values
- **param** penalty_: a penalty derived from the penalty class in penalty.h
- **param** smoothPenalty_: a smooth penalty derived from the smoothPenalty class in smoothPenalty.h
- **param** tuningParameters: tuning parameters for the penalty functions. Note that both penalty functions must
 take the same tuning parameters.
- **param** control_: settings for the glmnet optimizer.
- **return** fit result

### Version 2

- **T-param** nonsmoothPenalty: class of type nonsmoothPenalty (e.g., lasso, scad, lsp)
- **T-param** smoothPenalty: class of type smooth penalty (e.g., ridge)
- **T-param** tuning: tuning parameters used by both, the nonsmootPenalty and the smoothPenalty
- **param** model_: the model object derived from the model class in model.h
- **param** startingValues: an arma::rowvec vector with starting values
- **param** parameterLabels: a stringVector with parameter labels
- **param** penalty_: a penalty derived from the penalty class in penalty.h
- **param** smoothPenalty_: a smooth penalty derived from the smoothPenalty class in smoothPenalty.h
- **param** tuningParameters: tuning parameters for the penalty functions. Note that both penalty functions must
 take the same tuning parameters.
- **param** control_: settings for the glmnet optimizer.
- **return** fit result

## convergenceCriteriaGlmnet

- **value** GLMNET: Uses the convergence criterion outlined in Yuan et al. (2012) for GLMNET. Note that in case of BFGS, this will be identical to using the Armijo condition.
- **value** fitChange: Uses the change in fit from one iteration to the next.
- **value** gradients: Uses the gradients; if all are (close to) zero, the minimum is found

## controlDefaultGlmnet

Returns default for the optimizer settings

## Optimizer settings

The glmnet optimizer has the following additional settings:

- `initialHessian`: an `arma::mat` with the initial Hessian matrix fo the optimizer. In case of the simplified interface, this 
argument should not be used. Instead, pass the initial Hessian as shown above
- `stepSize`: a `double` specifying the initial stepSize of the outer iteration ($\theta_{k+1} = \theta_k + \text{stepSize} * \text{stepDirection}$)
- `sigma`: a `double` that is only relevant when lineSearch = 'GLMNET'. Controls the sigma parameter in Yuan, G.-X., Ho, C.-H., & Lin, C.-J. (2012). An improved GLMNET for l1-regularized logistic regression. The Journal of Machine Learning Research, 13, 1999–2030. https://doi.org/10.1145/2020408.2020421.
- `gamma`: a `double` controling the gamma parameter in Yuan, G.-X., Ho, C.-H., & Lin, C.-J. (2012). An improved GLMNET for l1-regularized logistic regression. The Journal of Machine Learning Research, 13, 1999–2030. https://doi.org/10.1145/2020408.2020421. Defaults to 0.
- `maxIterOut`: an `int` specifying the maximal number of outer iterations
- `maxIterIn`: an `int` specifying the maximal number of inner iterations
- `maxIterLine`: an `int` specifying the maximal number of iterations for the line search procedure
- `breakOuter`: a `double` specyfing the stopping criterion for outer iterations
- `breakInner`: a `double` specyfing the stopping criterion for inner iterations
- `convergenceCriterion`: a `convergenceCriteriaGlmnet` specifying which convergence criterion should be used for the outer iterations. Possible are `lessSEM::GLMNET`, `lessSEM::fitChange`,
and `lessSEM::gradients`. 
- `verbose`: an `int`, where 0 prints no additional information, > 0 prints GLMNET iterations

## Penalties

### CappedL1

#### tuningParametersCappedL1Glmnet

tuning parameters for the capped L1 penalty optimized with glmnet

- **param** weights: provide parameter-specific weights (e.g., for adaptive lasso)
- **param** lambda: lambda value >= 0
- **param** theta: theta value of the cappedL1 penalty > 0

#### penaltyCappedL1Glmnet

cappedL1 penalty for glmnet optimizer.


The penalty function is given by:
$$p( x_j) = \lambda \min(| x_j|, \theta)$$
where $\theta > 0$. The cappedL1 penalty is identical to the lasso for
parameters which are below $\theta$ and identical to a constant for parameters
above $\theta$. As adding a constant to the fitting function will not change its
minimum, larger parameters can stay unregularized while smaller ones are set to zero.

CappedL1 regularization:

* Zhang, T. (2010). Analysis of Multi-stage Convex Relaxation for Sparse Regularization.
Journal of Machine Learning Research, 11, 1081–1107.

### lasso

#### tuningParametersEnetGlmnet

Tuning parameters of the elastic net. For glmnet, we allow for different alphas and lambdas to combine penalties.p

- **param** lambda: parameter-specific lambda value >= 0
- **param** alpha: parameter-specific alpha value of the elastic net (relative importance of ridge and lasso)
- **param** weights: provide parameter-specific weights (e.g., for adaptive lasso)

#### penaltyLASSOGlmnet

lasso penalty for glmnet

The penalty function is given by:
$$p( x_j) = \lambda |x_j|$$
Lasso regularization will set parameters to zero if $\lambda$ is large enough

Lasso regularization:

* Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical
Society. Series B (Methodological), 58(1), 267–288.

### LSP

#### tuningParametersLspGlmnet

Tuning parameters for the lsp penalty optimized with glmnet


- **param** weights: provide parameter-specific weights (e.g., for adaptive lasso)
- **param** lambda: lambda value >= 0
- **param** theta: theta value of the lsp penalty > 0

#### penaltyLSPGlmnet

Lsp penalty for glmnet optimizer.
 
The penalty function is given by:
$$p( x_j) = \lambda \log(1 + |x_j|/\theta)$$
where $\theta > 0$.

lsp regularization:

* Candès, E. J., Wakin, M. B., & Boyd, S. P. (2008). Enhancing Sparsity by
Reweighted l1 Minimization. Journal of Fourier Analysis and Applications, 14(5–6),
877–905. https://doi.org/10.1007/s00041-008-9045-x


### MCP

#### tuningParametersMcpGlmnet

Tuning parameters for the mcp penalty optimized with glmnet

- **param** weights: provide parameter-specific weights (e.g., for adaptive lasso)
- **param** lambda: lambda value >= 0
- **param** theta: theta value of the cappedL1 penalty > 0

#### penaltyMcpGlmnet

Mcp penalty for glmnet optimizer

The penalty function is given by:
$$p( x_j) = \begin{cases}
\lambda |x_j| - x_j^2/(2\theta) & \text{if } |x_j| \leq \theta\lambda\\
\theta\lambda^2/2 & \text{if } |x_j| > \lambda\theta
\end{cases}$$
where $\theta > 1$.

mcp regularization:

* Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty.
The Annals of Statistics, 38(2), 894–942. https://doi.org/10.1214/09-AOS729


### Mixed Penalty

#### tuningParametersMixedGlmnet
 
 Tuning parameters for the mixed penalty optimized with glmnet

- **param** penaltyType_: penaltyType-vector specifying the penalty to be used for each parameter
- **param** lambda: provide parameter-specific lambda values
- **param** theta: theta value of the mixed penalty > 0
- **param** alpha: alpha value of the mixed penalty > 0
- **param** weights: provide parameter-specific weights (e.g., for adaptive lasso)

#### penaltyMixedGlmnet

Mixed penalty for glmnet optimizer

### Ridge

#### tuningParametersEnetGlmnet

Tuning parameters of the elastic net. For glmnet, we allow for different alphas and lambdas to combine penalties.p

- **param** lambda: parameter-specific lambda value >= 0
- **param** alpha: parameter-specific alpha value of the elastic net (relative importance of ridge and lasso)
- **param** weights: provide parameter-specific weights (e.g., for adaptive lasso)

#### penaltyRidgeGlmnet

ridge penalty for glmnet optimizer

The penalty function is given by:
$$p( x_j) = \lambda x_j^2$$
Note that ridge regularization will not set any of the parameters to zero
but result in a shrinkage towards zero.

Ridge regularization:

* Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation
for Nonorthogonal Problems. Technometrics, 12(1), 55–67.
https://doi.org/10.1080/00401706.1970.10488634

### SCAD

#### tuningParametersScadGlmnet

Tuning parameters for the scad penalty optimized with glmnet

- **param** weights: provide parameter-specific weights (e.g., for adaptive lasso)
- **param** lambda: lambda value >= 0
- **param** theta: theta value of the cappedL1 penalty > 0

#### penaltySCADGlmnet

Scad penalty for glmnet

The penalty function is given by:
$$p( x_j) = \begin{cases}
\lambda |x_j| & \text{if } |x_j| \leq \theta\\
\frac{-x_j^2 + 2\theta\lambda |x_j| - \lambda^2}{2(\theta -1)} &
\text{if } \lambda < |x_j| \leq \lambda\theta \\
(\theta + 1) \lambda^2/2 & \text{if } |x_j| \geq \theta\lambda\\
$$
where $\theta > 2$.

scad regularization:

* Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized
likelihood and its oracle properties. Journal of the American Statistical Association,
96(456), 1348–1360. https://doi.org/10.1198/016214501753382273
