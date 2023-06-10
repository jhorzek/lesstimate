# lesspar

> A longer documentation of the library can be found at https://jhorzek.github.io/lesspar/.

**lesspar** (**l**esspar **es**timates **s**parse **par**ameters) is a C++ header-only library that lets you combine statistical models such linear regression with state of the art penalty functions (e.g., lasso, elastic net, scad). With **lesspar** you can add regularization and variable selection procedures to your existing modeling framework. It is currently used in [**lessSEM**](https://github.com/jhorzek/lessSEM) to regularize structural equation models.

## Features 

- **Multiple penalty functions**: **lesspar** lets you apply any of the following penalties: ridge, lasso, adaptive lasso, elastic net, cappedL1, lsp, scad, mcp. Furthermore, you can combine multiple penalties.
- **State of the art optimizers**: **lesspar** provides two state of the art optimizers--variants of glmnet and ista.
- **Header-only**: **lesspar** is designed as a header-only library. Include the headers and you are ready to go.
- **Builds on armadillo**: **lesspar** builds on the popular *C++* [**armadillo**](https://arma.sourceforge.net/docs.html) library, providing you with access to a wide range of mathematical functions to create your model.
- **R and C++**: **lesspar** can be used in both, [*R*](https://github.com/jhorzek/lessparTemplateR) and [*C++*](https://github.com/jhorzek/lessparTemplateCpp) libraries.

## Details

**lesspar** lets you optimize fitting functions of the form

$$g(\pmb\theta) = f(\pmb\theta) + p(\pmb\theta),$$

where $f(\pmb\theta)$ is a smooth objective function (e.g., residual sum squared, weighted least squares or log-Likelihood) and $p(\pmb\theta)$ is a non-smooth penalty function (e.g., lasso or scad).

To use the optimziers, you will need two functions:

1. a function that computes the fit value $f(\pmb\theta)$ of your model
2. a functions that computes the gradients $\triangledown_{\pmb\theta}f(\pmb\theta)$ of the model

Given these two functions, **lesspar** lets you apply any of the aforementioned penalties with the quasi-Newton glmnet optimizer developed by Friedman et al. (2010) and Yuan et al. (2012) or variants of the proximal-operator based ista optimizer (see e.g., Gong et al., 2013). Because both optimziers provide a very similar interface, sitching between them is fairly simple. This interface is inspired by the [**ensmallen**](https://ensmallen.org/) library. 

A thorough introduction to **lesspar** and its use in R or C++ can be found in the [documentation](https://jhorzek.github.io/lesspar/). 
We also provide a [template for using **lesspar** in R](https://github.com/jhorzek/lessparTemplateR) and [template for using **lesspar** in C++](https://github.com/jhorzek/lessparTemplateCpp). Finally, you will find another example for including **lesspar** in R in the package [**lessLM**](https://github.com/jhorzek/lessLM). We recommend that you use the [simplified interfaces](https://github.com/jhorzek/lesspar/blob/main/include/simplified_interfaces.h) to get started. 

**lesspar** also stands for **Les**lie **s**nacks **par**ameters.  

## Example

The following code demonstrates the use of **lesspar** with regularized linear regressions. A longer step-by-step introduction including installation of **lesspar** is provided in the [documentation](https://jhorzek.github.io/lesspar/).

```
#include <armadillo>
#include "lesspar.h"

// The model must inherit from lesspar::model and override the fit and gradients function.
class linearRegressionModel : public lesspar::model
{
public:
    double fit(arma::rowvec b, lesspar::stringVector labels) override
    {
        // compute sum of squared errors
        arma::mat sse = (arma::trans(y - X * b.t()) * (y - X * b.t())) / (2.0 * y.n_elem);
        return (sse(0, 0));
    }

    arma::rowvec gradients(arma::rowvec b, lessSEM::stringVector labels) override
    {
        // compute gradients of sum of squared errors
        arma::rowvec grad = (arma::trans(-2.0 * X.t() * y + 2.0 * X.t() * X * b.t())) * (.5 / y.n_rows);
        return (grad);
    }

    // linear regression requires dependent variables y and design matrix X:
    const arma::colvec y;
    const arma::mat X;

    // constructor
    linearRegressionModel(arma::colvec y_, arma::mat X_) : y(y_), X(X_){};
};

int main()
{
    // examples for design matrix and dependent variable:
    arma::mat X = {{1.00, -0.70, -0.86},
                   {1.00, -1.20, -2.10},
                   {1.00, -0.15, 1.13},
                   {1.00, -0.50, -1.50},
                   {1.00, 0.83, 0.44},
                   {1.00, -1.52, -0.72},
                   {1.00, 1.40, -1.30},
                   {1.00, -0.60, -0.59},
                   {1.00, -1.10, 2.00},
                   {1.00, -0.96, -0.20}};

    arma::colvec y = {{0.56},
                      {-0.32},
                      {0.01},
                      {-0.09},
                      {0.18},
                      {-0.11},
                      {0.62},
                      {0.72},
                      {0.52},
                      {0.12}};

    // initialize model
    linearRegressionModel linReg(y, X);

    // To use the optimizers, you will need to
    // (1) specify starting values and names for the parameters
    arma::rowvec startingValues(3);
    startingValues.fill(0.0);
    std::vector<std::string> labels{"b0", "b1", "b2"};
    lessSEM::stringVector parameterLabels(labels);

    // (2) specify the penalty to be used for each of the parameters:
    std::vector<std::string> penalty{"none", "lasso", "lasso"};

    // (3) specify the tuning parameters of your penalty for
    // each parameter:
    arma::rowvec lambda = {{0.0, 0.2, 0.2}};
    // theta is not used by the lasso penalty:
    arma::rowvec theta = {{0.0, 0.0, 0.0}};

    lessSEM::fitResults fitResult_ = lessSEM::fitGlmnet(
        linReg,
        startingValues,
        parameterLabels,
        penalty,
        lambda,
        theta);
        
    return(0);
}
```

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

