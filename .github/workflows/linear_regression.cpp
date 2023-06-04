#include <armadillo>
#include <include/lessOptimizers.h>

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


// IMPORTANT: The library is calles lessOptimizers, but
// because it was initially a sub-folder of lessSEM, the
// namespace is still called lessSEM.

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
  //          Rcpp::StringVector. Otherwise it is a custom vector. that can
  //          be created with lessSEM::stringVector myVector(numberofParameters).
  // Second:
  //        1) fit must return a double (e.g., the -2-log-likelihood)
  //        2) gradients must return an arma::rowvec with the gradients. It is
  //           important that the gradients are returned in the same order as the
  //           parameters (i.e., don't shuffle your gradients, lessSEM will 
  //           assume that the first value in gradients corresponds to the
  //           derivative with respect to the first parameter passed to 
  //           the function).

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
  // sumSquaredError and sumSquaredErrorGradients function! To make these 
  // accessible to our functions, we have to define them:

  const arma::colvec y;
  const arma::mat X;

  // finally, we create a constructor for our class
  linearRegressionModel(arma::colvec y_, arma::mat X_) : y(y_), X(X_){};
};

int main()
{

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

    linearRegressionModel linReg(y, X);

    arma::rowvec startingValues(3);
    startingValues.fill(0.0);

    std::vector<std::string> labels{"b0", "b1", "b2"};
    lessSEM::stringVector parameterLabels(labels);

    // penalty: We don't penalize the intercept b0, but we penalize
    // b1 and b2 with lasso:
    std::vector<std::string> penalty{"none", "lasso", "lasso"};

    // tuning parameter lambda:
    arma::rowvec lambda = {{0.0, 0.2, 0.2}};
    // theta is not used by the lasso penalty:
    arma::rowvec theta = {{0.0, 0.0, 0.0}};

    lessSEM::fitResults fitResultGlmnet = lessSEM::fitGlmnet(
        linReg,
        startingValues,
        parameterLabels,
        penalty,
        lambda,
        theta //,
        // initialHessian, // optional, but can be very useful
    );

    std::cout << "### glmnet ###\n";
    std::cout << "fit: " << fitResultGlmnet.fit << "\n";
    std::cout << "parameters: " << fitResultGlmnet.parameterValues << "\n";

    std::cout << "### ista ###\n";
    lessSEM::fitResults fitResultIsta = lessSEM::fitIsta(
        linReg,
        startingValues,
        parameterLabels,
        penalty,
        lambda,
        theta);
    std::cout << "fit: " << fitResultIsta.fit << "\n";
    std::cout << "parameters: " << fitResultIsta.parameterValues << "\n";
}
