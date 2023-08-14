# ISTA

The implementation of ista follows that outlined in
Beck, A., & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding
Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences, 2(1),
183–202. https://doi.org/10.1137/080716542
see Remark 3.1 on p. 191 (ISTA with backtracking)

GIST can be found in
Gong, P., Zhang, C., Lu, Z., Huang, J., & Ye, J. (2013).
A General Iterative Shrinkage and Thresholding Algorithm for Non-convex
Regularized Optimization Problems. Proceedings of the 30th International
Conference on Machine Learning, 28(2)(2), 37–45.

## fitIsta

 We provide two optimizer interfaces: One uses a combination of arma::rowvec and less::stringVector for starting values and parameter labels respectively. This interface is consistent with the fit and gradient function of the `less::model`-class. Alternatively, a numericVector can be passed to the optimizers. This design is rooted in the use of Rcpp::NumericVectors that combine values and labels similar to an R vector. Thus, interfacing to this second function call can be easier when coming from R.

### Version 1

- **param** userModel: your model. Must inherit from less::model!
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
- **param** controlOptimizer: option to change the optimizer settings
- **param** verbose: should additional information be printed? If set > 0, additional
 information will be provided. Highly recommended for initial runs. Note that
 the optimizer itself has a separate verbose argument that can be used to print
 information on each iteration. This can be set with the controlOptimizer - argument.
- **return** fitResults

### Version 2

- **param** userModel: your model. Must inherit from less::model!
- **param** startingValues: an arma::rowvec numeric vector with starting values
- **param** parameterLabels: a less::stringVector with labels for parameters
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
- **param** controlOptimizer: option to change the optimizer settings
- **param** verbose: should additional information be printed? If set > 0, additional
information will be provided. Highly recommended for initial runs. Note that
the optimizer itself has a separate verbose argument that can be used to print
information on each iteration. This can be set with the controlOptimizer - argument.
- **return** fitResults

## ista

### Version 1

Implements (variants of) the ista optimizer.

- **param** model_: the model object derived from the model class in model.h
- **param** startingValuesRcpp: an Rcpp numeric vector with starting values
- **param** proximalOperator_: a proximal operator for the penalty function
- **param** penalty_: a penalty derived from the penalty class in penalty.h
- **param** smoothPenalty_: a smooth penalty derived from the smoothPenalty class in smoothPenalty.h
- **param** tuningParameters: tuning parameters for the penalty function
- **param** smoothTuningParameters: tuning parameters for the smooth penalty function
- **param** control_: settings for the ista optimizer. 
- **return** fit result

### Version 2

Implements (variants of) the ista optimizer.

- **param** model_: the model object derived from the model class in model.h
- **param** startingValues: an arma::rowvec numeric vector with starting values
- **param** parameterLabels: a less::stringVector with labels for parameters
- **param** proximalOperator_: a proximal operator for the penalty function
- **param** penalty_: a penalty derived from the penalty class in penalty.h
- **param** smoothPenalty_: a smooth penalty derived from the smoothPenalty class in smoothPenalty.h
- **param** tuningParameters: tuning parameters for the penalty function
- **param** smoothTuningParameters: tuning parameters for the smooth penalty function
- **param** control_: settings for the ista optimizer.
- **return** fit result

## controlDefault

Returns default for the optimizer settings

## Optimizer settings

The ista optimizer has the following additional settings:

- `L0`: a `double` controling the step size used in the first iteration
- `eta`: a `double` controling by how much the step size changes in inner iterations with $(\eta^i)*L$, where $i$ is the inner iteration
- `accelerate`: a `bool`; if  the extrapolation parameter is used to accelerate ista (see, e.g., Parikh, N., & Boyd, S. (2013). Proximal Algorithms. 
Foundations and Trends in Optimization, 1(3), 123–231., p. 152)
- `maxIterOut`: an `int` specifying the maximal number of outer iterations
- `maxIterIn`: an `int` specifying the maximal number of inner iterations
- `breakOuter`: a `double` specyfing the stopping criterion for outer iterations
- `breakInner`: a `double` specyfing the stopping criterion for inner iterations
- `convCritInner`: a `convCritInnerIsta` that specifies the inner breaking condition. Can be set to `less::istaCrit` (see Beck & Teboulle (2009);
 Remark 3.1 on p. 191 (ISTA with backtracking)) or `less::gistCrit` (see Gong et al., 2013; Equation 3) 
- `sigma`: a `double` in (0,1) that is used by the gist convergence criterion. Larger sigma enforce larger improvement in fit
- `stepSizeIn`: a `stepSizeInheritance` that specifies how step sizes should be carried forward from iteration to iteration. `less::initial`: resets the step size to L0 in each iteration, `less::istaStepInheritance`: takes the previous step size as initial value for the next iteration, `less::barzilaiBorwein`: uses the Barzilai-Borwein procedure, `less::stochasticBarzilaiBorwein`: uses the Barzilai-Borwein procedure, but sometimes resets the step size; this can help when the optimizer is caught in a bad spot.
- `sampleSize`: an `int` that can be used to scale the fitting function down if the fitting function depends on the sample size
- `verbose`: an `int`, where 0 prints no additional information, > 0 prints GLMNET iterations


### convCritInnerIsta

Convergence criteria used by the ista optimizer.

- **value** istaCrit: The approximated fit based on the quadratic approximation
h(parameters_k) := fit(parameters_k) +
(parameters_k-parameters_kMinus1)*gradients_k^T +
(L/2)*(parameters_k-parameters_kMinus1)^2 +
penalty(parameters_k)
is compared to the exact fit
- **value** gistCrit: the exact fit is compared to
h(parameters_k) := fit(parameters_k) +
penalty(parameters_kMinus1) +
L*(sigma/2)*(parameters_k-parameters_kMinus1)^2

### stepSizeInheritance

The ista optimizer provides different rules to be used to find an initial
step size. It defines if and how the step size should be carried forward
from iteration to iteration.

- **value** initial: resets the step size to L0 in each iteration
- **value** istaStepInheritance: takes the previous step size as initial value for the
next iteration
- **value** barzilaiBorwein: uses the Barzilai-Borwein procedure
- **value** stochasticBarzilaiBorwein: uses the Barzilai-Borwein procedure, but sometimes
resets the step size; this can help when the optimizer is caught in a bad spot.

## Penalties

### CappedL1

#### tuningParametersCappedL1

Tuning parameters for the cappedL1 penalty using ista

- **param** lambda: lambda value >= 0
- **param** alpha: alpha value of the elastic net (relative importance of ridge and lasso)
- **param** weights: provide parameter-specific weights (e.g., for adaptive lasso)
- **param** theta: threshold parameter; any parameter above this threshold will only receive the constant penalty lambda_i*theta, all below will get lambda_i*parameterValue_i

#### proximalOperatorCappedL1

Proximal operator for the cappedL1 penalty function

#### penaltyCappedL1

CappedL1 penalty for ista

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

#### tuningParametersEnet

Tuning parameters for the lasso penalty using ista

- **param** lambda: lambda value >= 0
- **param** alpha: alpha value of the elastic net (relative importance of ridge and lasso)
- **param** weights: provide parameter-specific weights (e.g., for adaptive lasso)

#### proximalOperatorLasso

Proximal operator for the lasso penalty function

#### penaltyLASSO

Lasso penalty for ista

The penalty function is given by:
$$p( x_j) = \lambda |x_j|$$
Lasso regularization will set parameters to zero if $\lambda$ is large enough

Lasso regularization:

* Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical
Society. Series B (Methodological), 58(1), 267–288.

### LSP

#### tuningParametersLSP

Tuning parameters for the lsp penalty using ista

- **param** weights: provide parameter-specific weights (e.g., for adaptive lasso)
- **param** lambda: lambda value >= 0
- **param** theta: theta value of the lsp penalty > 0

#### proximalOperatorLSP

Proximal operator for the lsp penalty function

#### penaltyLSP

Lsp penalty for ista

The penalty function is given by:
$$p( x_j) = \lambda \log(1 + |x_j|/\theta)$$
where $\theta > 0$.

lsp regularization:

* Candès, E. J., Wakin, M. B., & Boyd, S. P. (2008). Enhancing Sparsity by
Reweighted l1 Minimization. Journal of Fourier Analysis and Applications, 14(5–6),
877–905. https://doi.org/10.1007/s00041-008-9045-x

### MCP

#### tuningParametersMcp

Tuning parameters for the mcp penalty optimized with ista

- **param** weights: provide parameter-specific weights (e.g., for adaptive lasso)
- **param** lambda: lambda value >= 0
- **param** theta: theta value of the cappedL1 penalty > 0

#### proximalOperatorMcp

Proximal operator for the mcp penalty function

#### penaltyMcp

Mcp penalty for ista

The penalty function is given by:
$$p( x_j) = \begin{cases}
\lambda |x_j| - x_j^2/(2\theta) & \text{if } |x_j| \leq \theta\lambda\\
\theta\lambda^2/2 & \text{if } |x_j| > \lambda\theta
\end{cases}$$
where $\theta > 1$.

mcp regularization:

* Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty.
The Annals of Statistics, 38(2), 894–942. https://doi.org/10.1214/09-AOS729

### Mixed penalty

#### tuningParametersMixedPenalty
 
 Tuning parameters for the mixed penalty optimized with glmnet

- **param** penaltyType_: penaltyType-vector specifying the penalty to be used for each parameter
- **param** lambda: provide parameter-specific lambda values
- **param** theta: theta value of the mixed penalty > 0
- **param** alpha: alpha value of the mixed penalty > 0
- **param** weights: provide parameter-specific weights (e.g., for adaptive lasso)

#### proximalOperatorMixedPenalty

Proximal operator for the mixed penalty function

#### penaltyMixedPenalty

Mixed penalty

### Ridge

#### tuningParametersEnet

Tuning parameters for the lasso penalty using ista

- **param** lambda: lambda value >= 0
- **param** alpha: alpha value of the elastic net (relative importance of ridge and lasso)
- **param** weights: provide parameter-specific weights (e.g., for adaptive lasso)

#### penaltyRidge

Ridge penalty for ista

The penalty function is given by:
$$p( x_j) = \lambda x_j^2$$
Note that ridge regularization will not set any of the parameters to zero
but result in a shrinkage towards zero.

Ridge regularization:

* Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation
for Nonorthogonal Problems. Technometrics, 12(1), 55–67.
https://doi.org/10.1080/00401706.1970.10488634

### SCAD

#### tuningParametersScad

Tuning parameters for the scad penalty optimized with ista

- **param** weights: provide parameter-specific weights (e.g., for adaptive lasso)
- **param** lambda: lambda value >= 0
- **param** theta: theta value of the cappedL1 penalty > 0


#### proximalOperatorScad

Proximal operator for the scad penalty function

#### penaltyScad

Scad penalty for ista

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


  