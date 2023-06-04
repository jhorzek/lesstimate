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
