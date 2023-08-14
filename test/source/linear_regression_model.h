#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <armadillo>
#include <lesstimate.h>

inline double sumSquaredError(
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

inline arma::rowvec sumSquaredErrorGradients(
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

class linearRegressionModelNumericGradients : public less::model
{

public:
    // the less::model class has two methods: "fit" and "gradients".
    // Both of these methods must follow a fairly strict framework.
    // First: They must receive exactly two arguments:
    //        1) an arma::rowvec with current parameter values
    //        2) an Rcpp::StringVector with current parameter labels
    //          (NOTE: the lessSEM package currently does not make use of these
    //          labels. This is just for future use. If you don't want to use
    //          the labels, just pass any less::stringVector you want).
    //          if you are using R, a less::stringVector is just an
    //          Rcpp::StringVector. Otherwise it is a custom vector. that can
    //          be created with less::stringVector myVector(numberofParameters).
    // Second:
    //        1) fit must return a double (e.g., the -2-log-likelihood)
    //        2) gradients must return an arma::rowvec with the gradients. Here, we
    //           use automatically generated numerical gradients

    double fit(arma::rowvec b, less::stringVector labels) override
    {
        // NOTE: In sumSquaredError we assumed that b was a column-vector. We
        //  have to transpose b to make things work
        static_cast<void>(labels); // is unused
        return (sumSquaredError(b.t(), y, X));
    }

    // IMPORTANT: Note that we used some arguments above which we did not pass to
    // the functions: y, and X. Without these arguments, we cannot use our
    // sumSquaredError and sumSquaredErrorGradients function! To make these
    // accessible to our functions, we have to define them:

    const arma::colvec y;
    const arma::mat X;

    // finally, we create a constructor for our class
    linearRegressionModelNumericGradients(arma::colvec y_, arma::mat X_) : y(y_), X(X_){};
};

// IMPORTANT: The library is called lesstimate, but
// because it was initially a sub-folder of lessSEM, there
// are two namespaces that are identical: less and lessSEM.

class linearRegressionModel : public less::model
{

public:
    // the less::model class has two methods: "fit" and "gradients".
    // Both of these methods must follow a fairly strict framework.
    // First: They must receive exactly two arguments:
    //        1) an arma::rowvec with current parameter values
    //        2) an Rcpp::StringVector with current parameter labels
    //          (NOTE: the lessSEM package currently does not make use of these
    //          labels. This is just for future use. If you don't want to use
    //          the labels, just pass any less::stringVector you want).
    //          if you are using R, a less::stringVector is just an
    //          Rcpp::StringVector. Otherwise it is a custom vector. that can
    //          be created with less::stringVector myVector(numberofParameters).
    // Second:
    //        1) fit must return a double (e.g., the -2-log-likelihood)
    //        2) gradients must return an arma::rowvec with the gradients. It is
    //           important that the gradients are returned in the same order as the
    //           parameters (i.e., don't shuffle your gradients, lessSEM will
    //           assume that the first value in gradients corresponds to the
    //           derivative with respect to the first parameter passed to
    //           the function).

    double fit(arma::rowvec b, less::stringVector labels) override
    {
        static_cast<void>(labels); // currently not used; for later use
        // NOTE: In sumSquaredError we assumed that b was a column-vector. We
        //  have to transpose b to make things work
        return (sumSquaredError(b.t(), y, X));
    }

    arma::rowvec gradients(arma::rowvec b, less::stringVector labels) override
    {
        static_cast<void>(labels); // is unused
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

#endif