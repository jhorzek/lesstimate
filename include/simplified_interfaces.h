#ifndef SIMPLIFIED_INTERFACE
#define SIMPLIFIED_INTERFACE
// The optimizers implemented in lessOptimizers are fairly flexible, resulting 
// in a complexity overhead for users who just want to use one specific penalty 
// function out of the box. The following classes are meant to reduce this
// overhead by providing simplified interfaces to specific penalty functions.

#include <RcppArmadillo.h>
#include "glmnet_class.h"
#include "glmnet_penalties.h"

namespace lessSEM{

// resizeVector
// 
// We want to allow users to pass vectors of size 1 which are then resized to
// the correct lenght. The following two functions achieve this for std::vector
// and arma::rowvec.
// @param numberParameters target length of vector
// @param userVector user provided vector
// @return vector of length numberParameters, where, if userVector was of lenght 1,
// all elements are replaced with the single element provided by the user. Otherwise
// the vector is returned without any changes.
template<typename vecobj>
inline std::vector<vecobj> resizeVector(unsigned int numberParameters,
                                 std::vector<vecobj> userVector){
  if(userVector.size() == 1){
    vecobj userObj = userVector.at(0);
    userVector.resize(numberParameters);
    std::fill(userVector.begin(), userVector.end(), userObj);
  }
  return(userVector);
}

inline arma::rowvec resizeVector(unsigned int numberParameters,
                          arma::rowvec userVector){
  if(userVector.size() == 1){
    double userObj = userVector(0);
    userVector.resize(numberParameters);
    userVector.fill(userObj);
  }
  return(userVector);
}

// allEqual
// 
// checks if all elements of a vector with unsigned integers are the same.
// @param myvec vector with unsigned integers
// @return boolean indicating if all elements are the same.
inline bool allEqual(std::vector<unsigned int> myvec){
  if(myvec.size() == 0){
    Rcpp::stop("Empty vector");
  }
  unsigned int element_1 = myvec.at(0);
  for(auto i: myvec){
    if(i != element_1)
      return(false);
  }
  return(true);
}

// stringPenaltyToPenaltyType
//
// Translates a vector with strings to the internal penalty type representation.
// @param penalty string vector
// @return std::vector<penaltyType> with penalty types
inline std::vector<penaltyType> stringPenaltyToPenaltyType(std::vector<std::string> penalty){
  
  std::vector<penaltyType> penalties(penalty.size());
  
  for(unsigned int i = 0; i < penalty.size(); i++){
    
    if(penalty.at(i).compare("none") == 0){
      penalties.at(i) = penaltyType::none;
    }else if(penalty.at(i).compare("cappedL1") == 0){
      penalties.at(i) = penaltyType::cappedL1;
    }else if(penalty.at(i).compare("lasso") == 0){
      penalties.at(i) = penaltyType::lasso;
    } else if(penalty.at(i).compare("lsp") == 0){
      penalties.at(i) = penaltyType::lsp;
    }else if(penalty.at(i).compare("mcp") == 0){
      penalties.at(i) = penaltyType::mcp;
    }else if(penalty.at(i).compare("scad") == 0){
      penalties.at(i) = penaltyType::scad;
    }else{
      Rcpp::stop("Unknown penalty type: " + 
        penalty.at(i) + 
        ". Supported are: none, cappedL1, lasso, lsp, mcp, or scad."
      );
    }
  }
  
  return(penalties);
}

// printPenaltyDetails
// 
// prints information about the penalties if verbose is set to true.
// @param parameterLabels Rcpp::StringVector with the names of the parameters
// @param penalties penaltyType vector indicating the penalty for each parameter
// @param lambda lambda tuning parameter values. One lambda value for each parameter
// @param theta theta tuning parameter values. One theta value for each parameter
// @param verbose should information be printed
// @return nothing
inline void printPenaltyDetails(
    const Rcpp::StringVector& parameterLabels,
    const std::vector<penaltyType>& penalties,
    const arma::rowvec& lambda,
    const arma::rowvec& theta){
  
  std::vector<std::string> parameterLabels_(penalties.size());
  if((unsigned int)parameterLabels.size() != (unsigned int)penalties.size()){
    for(unsigned int i = 0; i < penalties.size(); i++){
      parameterLabels_.at(i) = std::to_string(i+1);
    }
  }else{
    for(unsigned int i = 0; i < penalties.size(); i++){
      parameterLabels_.at(i) = Rcpp::as<std::string>(parameterLabels.at(i));
    }
  }
  
  for(unsigned int i = 0; i < penalties.size(); i++){
    
    switch(penalties.at(i)){
    case none:
      Rcpp::Rcout << "No penalty on " 
                  << parameterLabels_.at(i)
                  << std::endl;
      break;
    case cappedL1:
      Rcpp::Rcout << "cappedL1 penalty on " 
                  << parameterLabels_.at(i)
                  << " lambda = " << lambda(i)
                  << " theta = " << theta(i)
                  << std::endl;
      break;
    case lasso:
      Rcpp::Rcout << "lasso penalty on " 
                  << parameterLabels_.at(i)
                  << " lambda = " << lambda(i)
                  << std::endl;
      break;
    case lsp:
      Rcpp::Rcout << "lsp penalty on " 
                  << parameterLabels_.at(i)
                  << " lambda = " << lambda(i)
                  << " theta = " << theta(i)
                  << std::endl;
      break;
    case mcp:
      Rcpp::Rcout << "mcp penalty on " 
                  << parameterLabels_.at(i)
                  << " lambda = " << lambda(i)
                  << " theta = " << theta(i)
                  << std::endl;
      break;
    case scad:
      Rcpp::Rcout << "scad penalty on " 
                  << parameterLabels_.at(i)
                  << " lambda = " << lambda(i)
                  << " theta = " << theta(i)
                  << std::endl;
      break;
    default:
      Rcpp::stop("Unknown penalty on " + parameterLabels_.at(i));
    }
    
  }
  
}


// glmnet
// 
// Function using defaults that proved to be reasonable when optimizing
// regularized SEM. Your mileage may vary, so please make sure to adapt the settings
// to your needs. 
// @param userModel your model. Must inherit from lessSEM::model!
// @param startingValues Rcpp::NumericVector with initial starting values. This 
// vector can have names.
// @param penalty vector with strings indicating the penalty for each parameter. 
// Currently supported are "none", "cappedL1", "lasso", "lsp", "mcp", and "scad". 
// (e.g., {"none", "scad", "scad", "lasso", "none"}). If only one value is provided,
// the same penalty will be applied to every parameter!
// @param lambda lambda tuning parameter values. One lambda value for each parameter. 
// If only one value is provided, this value will be applied to each parameter.
// Important: The the function will _not_ loop over these values but assume that you
// may want to provide different levels of regularization for each parameter!
// @param theta theta tuning parameter values. One theta value for each parameter
// If only one value is provided, this value will be applied to each parameter.
// Not all penalties use theta.
// Important: The the function will _not_ loop over these values but assume that you
// may want to provide different levels of regularization for each parameter!
// @param initialHessian matrix with initial Hessian values.
// @param control option to change the optimizer settings
// @param verbose should additional information be printed? If set > 0, additional
// information will be provided. Highly recommended for initial runs.
// @return fitResults
inline fitResults fitGlmnet(
    model& userModel,
    Rcpp::NumericVector startingValues,
    std::vector<std::string> penalty,
    arma::rowvec lambda,
    arma::rowvec theta,
    arma::mat initialHessian = arma::mat(1,1,arma::fill::ones),
    controlGLMNET control = controlGlmnetDefault(),
    const int verbose = 0
){
  
  unsigned int numberParameters = startingValues.length();
  Rcpp::StringVector parameterLabels = startingValues.names();
  
  // We expect startingValues, penalty, regularized, weights,
  // lambda, theta, and alpha to all be of the same length. For convenience, 
  // we will allow users to pass single values for each instead.
  penalty = resizeVector(numberParameters, penalty);
  lambda = resizeVector(numberParameters, lambda);
  theta = resizeVector(numberParameters, theta);
  
  auto penalties = stringPenaltyToPenaltyType(penalty);
  
  // resize Hessian if none is provided
  if((initialHessian.n_elem) == 1 && (numberParameters != 1)){
    double hessianValue = initialHessian(0,0);
    initialHessian.resize(numberParameters,numberParameters);
    initialHessian.fill(0.0);
    initialHessian.diag() += hessianValue;
    
    Rcpp::warning("Setting initial Hessian to identity matrix. We recommend passing a better Hessian.");
  }
  
  control.initialHessian = initialHessian;
  // check if all elements are of equal length now:
  std::vector<unsigned int> nElements{
    (unsigned int)penalties.size(),
    (unsigned int)lambda.n_elem,
    (unsigned int)theta.n_elem,
    (unsigned int)initialHessian.n_rows,
    (unsigned int)initialHessian.n_cols};
  
  if(!allEqual(nElements)
  ){
    Rcpp::stop("penalty, regularized, lambda, theta, alpha, nrow(initialHessian) and ncol(initialHessian) must all be of the same length.");
  }
  
  std::vector<double> weights(numberParameters);
  
  for(unsigned int i = 0; i < penalties.size(); i++){
    if(penalties.at(i) != penaltyType::none){
      weights.at(i) = 1.0;
    }else{
      weights.at(i) = 0.0;
    }
  }
  
  if(verbose)
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
  
  arma::rowvec parameters_k = startingValues;
  userModel.fit(parameters_k, parameterLabels);
  userModel.gradients(parameters_k, parameterLabels);
  
  fitResults fitResults_ = glmnet(
    userModel,
    startingValues,
    pen,
    smoothPen,
    tp,
    control
  );
  
  return(fitResults_);
}
}
#endif