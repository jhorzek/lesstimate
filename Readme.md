# lessOptimizers

This repository provides optimizers implemented in C++ header files for regularized model estimation in R using Rcpp. These optimizers are mainly used in the R package [**lessSEM**](https://github.com/jhorzek/lessSEM) to regularize structural equation models. The main objective of keeping a separate repository just for the optimizers is that they can also be used by other R packages by copying the include-folder in the inst-folder of your package or by keeping the files in a git submodule. This way you can make sure that your code will still work even if there are breaking changes in the optimizers used by **lessSEM**.

To use the optimziers, you will need two functions:

1. a function that computes the fit value (e.g., the -2log-Likelihood) of your model
2. a functions that computes the gradients of the model

Given these two functions, **lessOptimizers** lets you apply any of the following penalties to your model: ridge, lasso, adaptiveLasso, elasticNet, cappedL1, lsp, scad, or mcp. Models can be estimated using the glmnet optimizer or variants of the ista optimizer. The interface is inspired by the [**ensmallen**](https://ensmallen.org/) library. 

An introduction to optimizing models with **lessOptimizers** can be found in the [**lessLM**](https://github.com/jhorzek/lessLM) repository. We recommend that you use the [simplified interfaces](https://github.com/jhorzek/lessOptimizers/blob/main/include/simplified_interfaces.h) to get started. 

# References

## Penalty Functions

* Candès, E. J., Wakin, M. B., & Boyd, S. P. (2008). Enhancing Sparsity by 
Reweighted l1 Minimization. Journal of Fourier Analysis and Applications, 14(5–6), 
877–905. https://doi.org/10.1007/s00041-008-9045-x
* Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized 
likelihood and its oracle properties. Journal of the American Statistical 
Association, 96(456), 1348–1360. https://doi.org/10.1198/016214501753382273
* Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation 
for Nonorthogonal Problems. Technometrics, 12(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634
* Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. 
Journal of the Royal Statistical Society. Series B (Methodological), 58(1), 267–288.
* Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty. 
The Annals of Statistics, 38(2), 894–942. https://doi.org/10.1214/09-AOS729
* Zhang, T. (2010). Analysis of Multi-stage Convex Relaxation for Sparse Regularization. 
Journal of Machine Learning Research, 11, 1081–1107.
* Zou, H. (2006). The adaptive lasso and its oracle properties. Journal of the 
American Statistical Association, 101(476), 1418–1429. https://doi.org/10.1198/016214506000000735
* Zou, H., & Hastie, T. (2005). Regularization and variable selection via the 
elastic net. Journal of the Royal Statistical Society: Series B, 67(2), 301–320. 
https://doi.org/10.1111/j.1467-9868.2005.00503.x

## Optimizer

### GLMNET 

* Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for 
generalized linear models via coordinate descent. Journal of Statistical 
Software, 33(1), 1–20. https://doi.org/10.18637/jss.v033.i01
* Yuan, G.-X., Ho, C.-H., & Lin, C.-J. (2012). An improved GLMNET for 
l1-regularized logistic regression. The Journal of Machine Learning Research, 
13, 1999–2030. https://doi.org/10.1145/2020408.2020421

### Variants of ISTA

* Beck, A., & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding 
Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences, 2(1), 
183–202. https://doi.org/10.1137/080716542
* Gong, P., Zhang, C., Lu, Z., Huang, J., & Ye, J. (2013). A general iterative 
shrinkage and thresholding algorithm for non-convex regularized optimization problems. 
Proceedings of the 30th International Conference on Machine Learning, 28(2)(2), 37–45.
* Parikh, N., & Boyd, S. (2013). Proximal Algorithms. Foundations and 
Trends in Optimization, 1(3), 123–231.

# License

This project is under GPL >= 2.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

