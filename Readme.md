# lesspar

> A longer documentation of the library can be found at https://jhorzek.github.io/lesspar/.

**lesspar** (**l**esspar **es**timates **s**parse **par**ameters) is a C++ header-only library that lets you combine statistical models such linear regression with state of the art penalty functions (e.g., lasso, elastic net, scad). That is, with **lesspar** you can add regularization and variable selection procedures to your existing modeling framework.

## Features 

- **Header-only**: **lesspar** is designed as a header-only library. Adding **lesspar** to your existing project only requires including the header-files.
- **R and C++**: **lesspar** can be used in both, *R* and *C++* libraries. We provide templates for projects in [*R*](https://github.com/jhorzek/lessparTemplateR) and [*C++*](https://github.com/jhorzek/lessparTemplateCpp).
- **Builds on armadillo**: **lesspar** builds on the popular *C++* [**armadillo**](https://arma.sourceforge.net/docs.html) library, providing you with access to a wide range of mathematical functions to build your model.
- **Multiple penalty functions**: **lesspar** lets you apply any of the following penalties: **ridge**, **lasso**, **adaptive lasso**, **elastic net**, **cappedL1**, **lsp**, **scad**, **mcp**. Furthermore, you can combine multiple penalties.
- **State of the art optimizers**: **lesspar** provides two state of the art optimizers--variants of glmnet and ista.
- **Interface similar to ensmallen**: The optimizer interface is inspired by the optimizer library [**ensmallen**](https://ensmallen.org/). If you are already familiar with **ensmallen**, switching to **lesspar** is relatively easy.

## Details

**lesspar** lets you optimize fitting functions of the form

$$g(\pmb\theta) = f(\pmb\theta) + p(\pmb\theta),$$

where $f(\pmb\theta)$ is a smooth objective function (e.g., residual sum squared, weighted least squares or log-Likelihood) and $p(\pmb\theta)$ is a non-smooth penalty function (e.g., lasso or scad).

To use the optimziers, you will need two functions:

1. a function that computes the fit value $f(\pmb\theta)$ of your model
2. a functions that computes the gradients $\triangledown_{\pmb\theta}f(\pmb\theta)$ of the model

Given these two functions, **lesspar** lets you apply any of the following penalties: **ridge**, **lasso**, **adaptive lasso**, **elastic net**, **cappedL1**, **lsp**, **scad**, **mcp**, and mixtures therof. Currently two different optimizers are implemented:
glmnet is a quasi-Newton optimizers developed by Friedman et al. (2010) and Yuan et al. (2012). Ista is a proximal-operator based optimizer (see e.g., Gong et al., 2013). For smaller models, glmnet can be considerably faster than ista.
Because both optimziers provide a very similar interface, sitching between them is fairly simple.
This interface is inspired by the [**ensmallen**](https://ensmallen.org/) library. 

A thorough introduction to **lesspar** and its use in R or C++ can be found in the [documentation](https://jhorzek.github.io/lesspar/). 
We also provide a [template for using **lesspar** in R](https://github.com/jhorzek/lessparTemplateR) and [template for using **lesspar** in C++](https://github.com/jhorzek/lessparTemplateCpp). Finally, you will find another example for including **lesspar** in R in the package [**lessLM**](https://github.com/jhorzek/lessLM). We recommend that you use the [simplified interfaces](https://github.com/jhorzek/lesspar/blob/main/include/simplified_interfaces.h) to get started. 

**lesspar** also stands for **Les**lie **s**nacks **par**ameters.  

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
