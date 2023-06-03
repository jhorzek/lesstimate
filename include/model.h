#ifndef MODEL_H
#define MODEL_H

#include "common_headers.h"

namespace lessSEM{

// model
//
// model is the base class used in every optimizer implemented in lessOptimizers.
// The user specified model should inherit from the model class and must implement
// the two methods defined therein: 
// @method fit method with arguments parameterValues (arma::rowvec) and parameterLabels (stringVector; see common_headers.h)
// specifying the parameter values and the labels of the paramters. The function should return the fit value (double).
// @method gradients method with arguments parameterValues (arma::rowvec) and parameterLabels (stringVector; see common_headers.h)
// specifying the parameter values and the labels of the paramters. The function should return the gradients (arma::rowvec).
class model{
public:
  virtual double fit(arma::rowvec parameterValues,
                     stringVector parameterLabels) = 0;
  virtual arma::rowvec gradients(arma::rowvec parameterValues, 
                                 stringVector parameterLabels) = 0;
};

}
#endif