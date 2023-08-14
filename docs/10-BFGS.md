# BFGS

## bfgsOptim

Optimize a model using the BFGS optimizer. This optimizer does **not** support non-smooth penalty function (lasso, etc).

### Version 1:

- **param** model_: the model object derived from the model class in model.h
- **param** startingValuesRcpp: an Rcpp numeric vector with starting values
- **param** smoothPenalty_: a smooth penalty derived from the smoothPenalty class in smoothPenalty.h
- **param** tuningParameters: tuning parameters for the smoothPenalty function
- **param** control_: settings for the BFGS optimizer. Must be of struct controlBFGS. This can be created with controlBFGS.
- **return** fit result

### Version 2

- **T-param** T: type of the tuning parameters
- **param** model_: the model object derived from the model class in model.h
- **param** startingValues: an arma::rowvec numeric vector with starting values
- **param** parameterLabels: a lessSEM::stringVector with labels for parameters
- **param** smoothPenalty_: a smooth penalty derived from the smoothPenalty class in smoothPenalty.h
- **param** tuningParameters: tuning parameters for the smoothPenalty function
- **param** control_: settings for the BFGS optimizer. Must be of struct controlBFGS. This can be created with controlBFGS.
- **return** fit result

## controlBFGS

Struct that allows you to adapt the optimizer settings for the BFGS optimizer.

- **param** initialHessian: initial Hessian matrix fo the optimizer.
- **param** stepSize: Initial stepSize of the outer iteration (theta_{k+1} = theta_k + stepSize * Stepdirection)
- **param** sigma: only relevant when lineSearch = 'GLMNET'. Controls the sigma parameter in Yuan, G.-X., Ho, C.-H., & Lin, C.-J. (2012). An improved GLMNET for l1-regularized logistic regression. The Journal of Machine Learning Research, 13, 1999–2030. https:*doi.org/10.1145/2020408.2020421.
- **param** gamma: Controls the gamma parameter in Yuan, G.-X., Ho, C.-H., & Lin, C.-J. (2012). An improved GLMNET for l1-regularized logistic regression. The Journal of Machine Learning Research, 13, 1999–2030. https://doi.org/10.1145/2020408.2020421. Defaults to 0.
- **param** maxIterOut: Maximal number of outer iterations
- **param** maxIterIn: Maximal number of inner iterations
- **param** maxIterLine: Maximal number of iterations for the line search procedure
- **param** breakOuter: Stopping criterion for outer iterations
- **param** breakInner: Stopping criterion for inner iterations
- **param** convergenceCriterion: which convergence criterion should be used for the outer iterations? possible are 0 = GLMNET, 1 = fitChange, 2 = gradients.
 Note that in case of gradients and GLMNET, we divide the gradients (and the Hessian) of the log-Likelihood by N as it would otherwise be
 considerably more difficult for larger sample sizes to reach the convergence criteria.
- **param** verbose: 0 prints no additional information, > 0 prints GLMNET iterations



