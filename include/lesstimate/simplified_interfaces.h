#ifndef SIMPLIFIED_INTERFACE
#define SIMPLIFIED_INTERFACE
// The optimizers implemented in lesstimate are fairly flexible, resulting
// in a complexity overhead for users who just want to use one specific penalty
// function out of the box. The following classes are meant to reduce this
// overhead by providing simplified interfaces to specific penalty functions.

#include "simplified_interfaces_helper.h"

namespace lessSEM
{

  // We provide two optimizer interfaces: One uses a combination of arma::rowvec and lessSEM::stringVector for starting
  // values and parameter labels respectively. This interface is consistent with the fit and gradient function of the
  // lessSEM::model-class. Alternatively, a numericVector can be passed to the optimizers. This design is rooted in
  // the use of Rcpp::NumericVectors that combine values and labels similar to an R vector. Thus, interfacing to this
  // second function call can be easier when coming from R.

  /**
   * @brief Function using defaults that proved to be reasonable when optimizing
  * regularized SEM. Your mileage may vary, so please make sure to adapt the settings
  * to your needs.
  * @param userModel your model. Must inherit from lessSEM::model!
  * @param startingValues numericVector with initial starting values. This
  * vector can have names.
  * @param penalty vector with strings indicating the penalty for each parameter.
  * Currently supported are "none", "cappedL1", "lasso", "lsp", "mcp", and "scad".
  * (e.g., {"none", "scad", "scad", "lasso", "none"}). If only one value is provided,
  * the same penalty will be applied to every parameter!
  * @param lambda lambda tuning parameter values. One lambda value for each parameter.
  * If only one value is provided, this value will be applied to each parameter.
  * Important: The the function will _not_ loop over these values but assume that you
  * may want to provide different levels of regularization for each parameter!
  * @param theta theta tuning parameter values. One theta value for each parameter
  * If only one value is provided, this value will be applied to each parameter.
  * Not all penalties use theta.
  * Important: The the function will _not_ loop over these values but assume that you
  * may want to provide different levels of regularization for each parameter!
  * @param initialHessian matrix with initial Hessian values.
  * @param controlOptimizer option to change the optimizer settings
  * @param verbose should additional information be printed? If set > 0, additional
  * information will be provided. Highly recommended for initial runs. Note that
  * the optimizer itself has a separate verbose argument that can be used to print
  * information on each iteration. This can be set with the controlOptimizer - argument.
  * @return fitResults
  */
  inline fitResults fitGlmnet(
      model &userModel,
      numericVector startingValues,
      std::vector<std::string> penalty,
      arma::rowvec lambda,
      arma::rowvec theta,
      arma::mat initialHessian = arma::mat(1, 1, arma::fill::ones),
      controlGLMNET controlOptimizer = controlGlmnetDefault(),
      const int verbose = 0)
  {

    unsigned int numberParameters = startingValues.length();
    stringVector parameterLabels = startingValues.names();

    // We expect startingValues, penalty, regularized, weights,
    // lambda, theta, and alpha to all be of the same length. For convenience,
    // we will allow users to pass single values for each instead.
    penalty = resizeVector(numberParameters, penalty);
    lambda = resizeVector(numberParameters, lambda);
    theta = resizeVector(numberParameters, theta);

    auto penalties = stringPenaltyToPenaltyType(penalty);

    // resize Hessian if none is provided
    if ((initialHessian.n_elem) == 1 && (numberParameters != 1))
    {
      double hessianValue = initialHessian(0, 0);
      initialHessian.resize(numberParameters, numberParameters);
      initialHessian.fill(0.0);
      initialHessian.diag() += hessianValue;

      warn("Setting initial Hessian to identity matrix. We recommend passing a better Hessian.");
    }

    controlOptimizer.initialHessian = initialHessian;
    // check if all elements are of equal length now:
    std::vector<unsigned int> nElements{
        (unsigned int)penalties.size(),
        (unsigned int)lambda.n_elem,
        (unsigned int)theta.n_elem,
        (unsigned int)initialHessian.n_rows,
        (unsigned int)initialHessian.n_cols};

    if (!allEqual(nElements))
    {
      error("penalty, regularized, lambda, theta, alpha, nrow(initialHessian) and ncol(initialHessian) must all be of the same length.");
    }

    std::vector<double> weights(numberParameters);

    for (unsigned int i = 0; i < penalties.size(); i++)
    {
      if (penalties.at(i) != penaltyType::none)
      {
        weights.at(i) = 1.0;
      }
      else
      {
        weights.at(i) = 0.0;
      }
    }

    if (verbose)
      printPenaltyDetails(
          parameterLabels,
          penalties,
          lambda,
          theta);

    tuningParametersMixedGlmnet tp;
    tp.alpha = arma::rowvec(numberParameters, arma::fill::ones);
    tp.lambda = lambda;
    tp.penaltyType_ = penalties;
    tp.theta = theta;
    tp.weights = weights;

    penaltyMixedGlmnet pen;
    noSmoothPenalty<tuningParametersMixedGlmnet> smoothPen;

    // optimize

    fitResults fitResults_ = glmnet(
        userModel,
        startingValues,
        pen,
        smoothPen,
        tp,
        controlOptimizer);

    return (fitResults_);
  }
  /**
   * @brief Function using defaults that proved to be reasonable when optimizing
  * regularized SEM. Your mileage may vary, so please make sure to adapt the settings
  * to your needs.
  * 
  * @param userModel your model. Must inherit from lessSEM::model!
  * @param startingValues an arma::rowvec numeric vector with starting values
  * @param parameterLabels a lessSEM::stringVector with labels for parameters
  * @param penalty vector with strings indicating the penalty for each parameter.
  * Currently supported are "none", "cappedL1", "lasso", "lsp", "mcp", and "scad".
  * (e.g., {"none", "scad", "scad", "lasso", "none"}). If only one value is provided,
  * the same penalty will be applied to every parameter!
  * @param lambda lambda tuning parameter values. One lambda value for each parameter.
  * If only one value is provided, this value will be applied to each parameter.
  * Important: The the function will _not_ loop over these values but assume that you
  * may want to provide different levels of regularization for each parameter!
  * @param theta theta tuning parameter values. One theta value for each parameter
  * If only one value is provided, this value will be applied to each parameter.
  * Not all penalties use theta.
  * Important: The the function will _not_ loop over these values but assume that you
  * may want to provide different levels of regularization for each parameter!
  * @param initialHessian matrix with initial Hessian values.
  * @param controlOptimizer option to change the optimizer settings
  * @param verbose should additional information be printed? If set > 0, additional
  * information will be provided. Highly recommended for initial runs. Note that
  * the optimizer itself has a separate verbose argument that can be used to print
  * information on each iteration. This can be set with the controlOptimizer - argument.
  * @return fitResults
  */
  inline fitResults fitGlmnet(
      model &userModel,
      arma::rowvec startingValues,
      stringVector parameterLabels,
      std::vector<std::string> penalty,
      arma::rowvec lambda,
      arma::rowvec theta,
      arma::mat initialHessian = arma::mat(1, 1, arma::fill::ones),
      controlGLMNET controlOptimizer = controlGlmnetDefault(),
      const int verbose = 0)
  {
    numericVector startingValuesNumVec = toNumericVector(startingValues);
    startingValuesNumVec.names() = parameterLabels;

    return (fitGlmnet(
        userModel,
        startingValuesNumVec,
        penalty,
        lambda,
        theta,
        initialHessian,
        controlOptimizer,
        verbose));
  }

  // ISTA OPTIMIZER

  // We provide two optimizer interfaces: One uses a combination of arma::rowvec and lessSEM::stringVector for starting
  // values and parameter labels respectively. This interface is consistent with the fit and gradient function of the
  // lessSEM::model-class. Alternatively, a numericVector can be passed to the optimizers. This design is rooted in
  // the use of Rcpp::NumericVectors that combine values and labels similar to an R vector. Thus, interfacing to this
  // second function call can be easier when coming from R.

/**
 * @brief Function using defaults that proved to be reasonable when optimizing
  * regularized SEM. Your mileage may vary, so please make sure to adapt the settings
  * to your needs.
  * @param userModel your model. Must inherit from lessSEM::model!
  * @param startingValues numericVector with initial starting values. This
  * vector can have names.
  * @param penalty vector with strings indicating the penalty for each parameter.
  * Currently supported are "none", "cappedL1", "lasso", "lsp", "mcp", and "scad".
  * (e.g., {"none", "scad", "scad", "lasso", "none"}). If only one value is provided,
  * the same penalty will be applied to every parameter!
  * @param lambda lambda tuning parameter values. One lambda value for each parameter.
  * If only one value is provided, this value will be applied to each parameter.
  * Important: The the function will _not_ loop over these values but assume that you
  * may want to provide different levels of regularization for each parameter!
  * @param theta theta tuning parameter values. One theta value for each parameter
  * If only one value is provided, this value will be applied to each parameter.
  * Not all penalties use theta.
  * Important: The the function will _not_ loop over these values but assume that you
  * may want to provide different levels of regularization for each parameter!
  * @param controlOptimizer option to change the optimizer settings
  * @param verbose should additional information be printed? If set > 0, additional
  * information will be provided. Highly recommended for initial runs. Note that
  * the optimizer itself has a separate verbose argument that can be used to print
  * information on each iteration. This can be set with the controlOptimizer - argument.
  * @return fitResults
  */
  inline fitResults fitIsta(
      model &userModel,
      numericVector startingValues,
      std::vector<std::string> penalty,
      arma::rowvec lambda,
      arma::rowvec theta,
      controlIsta controlOptimizer = controlIstaDefault(),
      const int verbose = 0)
  {

    unsigned int numberParameters = startingValues.length();
    stringVector parameterLabels = startingValues.names();

    // We expect startingValues, penalty, regularized, weights,
    // lambda, theta, and alpha to all be of the same length. For convenience,
    // we will allow users to pass single values for each instead.
    penalty = resizeVector(numberParameters, penalty);
    lambda = resizeVector(numberParameters, lambda);
    theta = resizeVector(numberParameters, theta);

    auto penalties = stringPenaltyToPenaltyType(penalty);

    // check if all elements are of equal length now:
    std::vector<unsigned int> nElements{
        (unsigned int)penalties.size(),
        (unsigned int)lambda.n_elem,
        (unsigned int)theta.n_elem};

    if (!allEqual(nElements))
    {
      error("penalty, regularized, lambda, theta, and alpha must all be of the same length.");
    }

    std::vector<double> weights(numberParameters);

    for (unsigned int i = 0; i < penalties.size(); i++)
    {
      if (penalties.at(i) != penaltyType::none)
      {
        weights.at(i) = 1.0;
      }
      else
      {
        weights.at(i) = 0.0;
      }
    }

    if (verbose)
      printPenaltyDetails(
          parameterLabels,
          penalties,
          lambda,
          theta);

    tuningParametersMixedPenalty tp;
    tp.alpha = arma::rowvec(numberParameters, arma::fill::ones);
    tp.lambda = lambda;
    tp.pt = penalties;
    tp.theta = theta;
    tp.weights = weights;

    tuningParametersEnet smoothTp;
    smoothTp.alpha = 0.0;
    smoothTp.lambda = 0.0;
    smoothTp.weights = weights;

    proximalOperatorMixedPenalty proximalOperatorMixedPenalty_;
    penaltyMixedPenalty penalty_;
    penaltyRidge smoothPenalty_;

    // optimize

    fitResults fitResults_ = ista(
        userModel,
        startingValues,
        proximalOperatorMixedPenalty_,
        penalty_,
        smoothPenalty_,
        tp,
        smoothTp,
        controlOptimizer);

    return (fitResults_);
  }

/**
 * @brief Function using defaults that proved to be reasonable when optimizing
  * regularized SEM. Your mileage may vary, so please make sure to adapt the settings
  * to your needs.
  * @param userModel your model. Must inherit from lessSEM::model!
  * @param startingValues an arma::rowvec numeric vector with starting values
  * @param parameterLabels a lessSEM::stringVector with labels for parameters
  * @param penalty vector with strings indicating the penalty for each parameter.
  * Currently supported are "none", "cappedL1", "lasso", "lsp", "mcp", and "scad".
  * (e.g., {"none", "scad", "scad", "lasso", "none"}). If only one value is provided,
  * the same penalty will be applied to every parameter!
  * @param lambda lambda tuning parameter values. One lambda value for each parameter.
  * If only one value is provided, this value will be applied to each parameter.
  * Important: The the function will _not_ loop over these values but assume that you
  * may want to provide different levels of regularization for each parameter!
  * @param theta theta tuning parameter values. One theta value for each parameter
  * If only one value is provided, this value will be applied to each parameter.
  * Not all penalties use theta.
  * Important: The the function will _not_ loop over these values but assume that you
  * may want to provide different levels of regularization for each parameter!
  * @param controlOptimizer option to change the optimizer settings
  * @param verbose should additional information be printed? If set > 0, additional
  * information will be provided. Highly recommended for initial runs. Note that
  * the optimizer itself has a separate verbose argument that can be used to print
  * information on each iteration. This can be set with the controlOptimizer - argument.
  * @return fitResults
  */
  inline fitResults fitIsta(
      model &userModel,
      arma::rowvec startingValues,
      stringVector parameterLabels,
      std::vector<std::string> penalty,
      arma::rowvec lambda,
      arma::rowvec theta,
      controlIsta controlOptimizer = controlIstaDefault(),
      const int verbose = 0)
  {
    numericVector startingValuesNumVec = toNumericVector(startingValues);
    startingValuesNumVec.names() = parameterLabels;

    return (
        fitIsta(
            userModel,
            startingValuesNumVec,
            penalty,
            lambda,
            theta,
            controlOptimizer,
            verbose));
  }
}
#endif