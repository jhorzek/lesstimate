# Getting Started

To use **lessOptimizers** for your project, you will need two functions.
First, a function that computes $f(\pmb\theta)$, the value of your un-
regularized objective function. Second, a function that computes 
$\triangledown_{\pmb\theta} f(\pmb\theta)$, the gradient vector of your
un-regularized objective function with respect to the parameters $\pmb\theta$.
We also assume that these functions use armadillo. If you don't use armadillo,
you may have to write a translation layer.

Let's assume that you want to estimate a linear regression model. In this case,
these functions could look as follows:

```
#include <armadillo>

double sumSquaredError(
    arma::colvec b, // the parameter vector
    arma::colvec y, // the dependent variable
    arma::mat X     // the design matrix
)
{
  // compute the sum of squared errors:
  arma::mat sse = arma::trans(y - X * b) * (y - X * b);

  // other packages, such as glmnet, scale the sse with
  // 1/(2*N), where N is the sample size. We will do that here as well

  sse *= 1.0 / (2.0 * y.n_elem);

  // note: We must return a double, but the sse is a matrix
  // To get a double, just return the single value that is in
  // this matrix:
  return (sse(0, 0));
}

arma::rowvec sumSquaredErrorGradients(
    arma::colvec b, // the parameter vector
    arma::colvec y, // the dependent variable
    arma::mat X     // the design matrix
)
{
  // note: we want to return our gradients as row-vector; therefore,
  // we have to transpose the resulting column-vector:
  arma::rowvec gradients = arma::trans(-2.0 * X.t() * y + 2.0 * X.t() * X * b);

  // other packages, such as glmnet, scale the sse with
  // 1/(2*N), where N is the sample size. We will do that here as well

  gradients *= (.5 / y.n_rows);

  return (gradients);
 }

```

With these two functions, we are ready to go. If you want to use the glmnet optimizer,
you may want to also implement a function that computes the Hessian. It can be beneficial to
provide a good starting point for the bfgs updates using this Hessian.


## Step 1: Creating a model object

**lessOptimizers** assumes that you pass a model-object to the optimizers. This model 
object ist implemented in the [`lessSEM::model`-class] ADD LINK TO FILE IN GITHUB.
Therefore, we have to create a custom class that inherits from `lessSEM::model` and 
implements our linear regression using the functions defined above:

```
class linearRegressionModel : public lessSEM::model
{

public:
  // the lessSEM::model class has two methods: "fit" and "gradients".
  // Both of these methods must follow a fairly strict framework.
  // First: They must receive exactly two arguments:
  //        1) an arma::rowvec with current parameter values
  //        2) an Rcpp::StringVector with current parameter labels
  //          (NOTE: the lessSEM package currently does not make use of these
  //          labels. This is just for future use. If you don't want to use 
  //          the labels, just pass any lessSEM::stringVector you want).
  //          if you are using R, a lessSEM::stringVector is just an 
  //	      Rcpp::StringVector. Otherwise it is a custom vector. that can
  //	      be created with lessSEM::stringVector myVector(numberofParameters).
  // Second:
  //        1) fit must return a double (e.g., the -2-log-likelihood)
  //        2) gradients must return an arma::rowvec with the gradients. It is
  //           important that the gradients are returned in the same order as the
  //           parameters (i.e., don't shuffle your gradients, lessSEM will assume
  //           that the first value in gradients corresponds to the derivative with
  //           respect to the first parameter passed to the function).

  double fit(arma::rowvec b, lessSEM::stringVector labels) override
  {
    // NOTE: In sumSquaredError we assumed that b was a column-vector. We
    //  have to transpose b to make things work
    return (sumSquaredError(b.t(), y, X));
  }

  arma::rowvec gradients(arma::rowvec b, lessSEM::stringVector labels) override
  {
    // NOTE: In sumSquaredErrorGradients we assumed that b was a column-vector. We
    //  have to transpose b to make things work
    return (sumSquaredErrorGradients(b.t(), y, X));
  }

  // IMPORTANT: Note that we used some arguments above which we did not pass to
  // the functions: y, and X. Without these arguments, we cannot use our
  // sumSquaredError and sumSquaredErrorGradients function! To make these accessible
  // to our functions, we have to define them:

  const arma::colvec y;
  const arma::mat X;

  // finally, we create a constructor for our class
  linearRegressionModel(arma::colvec y_, arma::mat X_) : y(y_), X(X_){};
};

```

Instances of `linearRegressionModel` can be passed to the `glmnet` or `ista` optimizers.

## Step 2: Interfacing to the optimizers

Ther are two interfaces you can use: 

1. A specialized interface, where the model is penalized only using one specific penalty function.
This requires more work, but is typically a bit faster than using the second approach.
2. A simplified interface that allows you to use any of the penalty functions (and also mix them)

We will use the simplified interface in the following. To this end, we will first create a new
instance of our linearRegressionModel:

```
arma::mat X

linearRegressionModel linReg(y, X);
```

Next, we create a vector with starting values using **armadillo**. This vector must be of 
length numberofParameters, the number of parameters in the model.

```
numberOfParameters = X.n_cols;
arma::rowvec startingValues(n
```



**lessOptimizers** is inspired by the 
