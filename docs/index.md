# lesstimate

**lesstimate** (**l**esstimate **es**timates **s**parse es**timate**s) is a C++ header-only library that lets you combine statistical models such linear regression with state of the art penalty functions (e.g., lasso, elastic net, scad). With **lesstimate** you can add regularization and variable selection procedures to your existing modeling framework. It is currently used in [**lessSEM**](https://github.com/jhorzek/lessSEM) to regularize structural equation models.

## Features 

- **Multiple penalty functions**: **lesstimate** lets you apply any of the following penalties: ridge, lasso, adaptive lasso, elastic net, cappedL1, lsp, scad, mcp. Furthermore, you can combine multiple penalties.
- **State of the art optimizers**: **lesstimate** provides two state of the art optimizers--variants of glmnet and ista.
- **Header-only**: **lesstimate** is designed as a header-only library. Include the headers and you are ready to go.
- **Builds on armadillo**: **lesstimate** builds on the popular *C++* [**armadillo**](https://arma.sourceforge.net/docs.html) library, providing you with access to a wide range of mathematical functions to create your model.
- **R and C++**: **lesstimate** can be used in both, [*R*](https://github.com/jhorzek/lesstimateTemplateR) and [*C++*](https://github.com/jhorzek/lesstimateTemplateCpp) libraries.

![peanlty functions](penaltyFunctions.png)

## Details

**lesstimate** lets you optimize fitting functions of the form

$$g(\pmb\theta) = f(\pmb\theta) + p(\pmb\theta),$$

where $f(\pmb\theta)$ is a smooth objective function (e.g., residual sum squared, weighted least squares or log-Likelihood) and $p(\pmb\theta)$ is a non-smooth penalty function (e.g., lasso or scad).

To use the optimziers, you will need two functions:

1. a function that computes the fit value $f(\pmb\theta)$ of your model
2. (optional) a functions that computes the gradients $\triangledown_{\pmb\theta}f(\pmb\theta)$ of the model. If none is provided, the gradients will be approximated numerically, which can be quite slow

Given these two functions, **lesstimate** lets you apply any of the aforementioned penalties with the quasi-Newton glmnet optimizer developed by Friedman et al. (2010) and Yuan et al. (2012) or variants of the proximal-operator based ista optimizer (see e.g., Gong et al., 2013). Because both optimziers provide a very similar interface, switching between them is fairly simple. This interface is inspired by the [**ensmallen**](https://ensmallen.org/) library. 

**lesstimate** was mainly developed to be used in [**lessSEM**](https://jhorzek.github.io/lessSEM/), an 
R package for regularized Structural Equation Models. However, the library can also be used from C++.
**lesstimate** builds heavily on the **RcppAdmadillo** (Eddelbuettel et al., 2014) and **armadillo** (Sanderson et al., 2016) 
libraries. The optimizer interface is inspired by the **ensmallen** library (Curtin et al., 2021).

# References

## Software

- Curtin R R, Edel M, Prabhu R G, Basak S, Lou Z, Sanderson C (2021). The ensmallen library for flexible numerical optimization. Journal of Machine Learning Research, 22 (166).
- Eddelbuettel D, Sanderson C (2014). “RcppArmadillo: Accelerating R with high-performance C++ linear algebra.” Computational Statistics and Data Analysis, 71, 1054–1063. doi:10.1016/j.csda.2013.02.005.
- Sanderson C, Curtin R (2016). Armadillo: a template-based C++ library for linear algebra. Journal of Open Source Software, 1 (2), pp. 26.

## Penalty Functions

- Candès, E. J., Wakin, M. B., & Boyd, S. P. (2008). Enhancing Sparsity
  by Reweighted l1 Minimization. Journal of Fourier Analysis and
  Applications, 14(5–6), 877–905.
  <https://doi.org/10.1007/s00041-008-9045-x>
- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized
  likelihood and its oracle properties. Journal of the American
  Statistical Association, 96(456), 1348–1360.
  <https://doi.org/10.1198/016214501753382273>
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased
  Estimation for Nonorthogonal Problems. Technometrics, 12(1), 55–67.
  <https://doi.org/10.1080/00401706.1970.10488634>
- Tibshirani, R. (1996). Regression shrinkage and selection via the
  lasso. Journal of the Royal Statistical Society. Series B
  (Methodological), 58(1), 267–288.
- Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax
  concave penalty. The Annals of Statistics, 38(2), 894–942.
  <https://doi.org/10.1214/09-AOS729>
- Zhang, T. (2010). Analysis of Multi-stage Convex Relaxation for Sparse
  Regularization. Journal of Machine Learning Research, 11, 1081–1107.
- Zou, H. (2006). The adaptive lasso and its oracle properties. Journal
  of the American Statistical Association, 101(476), 1418–1429.
  <https://doi.org/10.1198/016214506000000735>
- Zou, H., & Hastie, T. (2005). Regularization and variable selection
  via the elastic net. Journal of the Royal Statistical Society: Series
  B, 67(2), 301–320. <https://doi.org/10.1111/j.1467-9868.2005.00503.x>


## GLMNET

- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of Statistical Software, 33(1), 1–20. https://doi.org/10.18637/jss.v033.i01
- Yuan, G.-X., Ho, C.-H., & Lin, C.-J. (2012). An improved GLMNET for l1-regularized logistic regression. The Journal of Machine Learning Research, 13, 1999–2030. https://doi.org/10.1145/2020408.2020421

## Variants of ISTA

- Beck, A., & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences, 2(1), 183–202. https://doi.org/10.1137/080716542
- Gong, P., Zhang, C., Lu, Z., Huang, J., & Ye, J. (2013). A general iterative shrinkage and thresholding algorithm for non-convex regularized optimization problems. Proceedings of the 30th International Conference on Machine Learning, 28(2)(2), 37–45.
- Parikh, N., & Boyd, S. (2013). Proximal Algorithms. Foundations and Trends in Optimization, 1(3), 123–231.

