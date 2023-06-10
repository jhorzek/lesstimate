#ifndef MODEL_H
#define MODEL_H

#include "common_headers.h"

namespace lessSEM
{
  /**
   * @brief model is the base class used in every optimizer implemented in lesspar.
   * The user specified model should inherit from the model class and must implement
   * the two methods defined therein
   */
  class model
  {
  public:
    /**
     * @brief fit method  with arguments parameterValues (arma::rowvec) and parameterLabels (stringVector; see common_headers.h)
     * specifying the parameter values and the labels of the paramters. The function should return the fit value (double).
     *
     * @param parameterValues numericVector with parameter values
     * @param parameterLabels stringVector with parameterLabels
     * @return double
     */
    virtual double fit(arma::rowvec parameterValues,
                       stringVector parameterLabels) = 0;

    /**
     * @brief gradients method with arguments parameterValues(arma::rowvec) and parameterLabels(stringVector; see common_headers.h) * specifying the parameter values and the labels of the paramters.The function should return the gradients(arma::rowvec)
     *
     * @param parameterValues numericVector with parameter values
     * @param parameterLabels stringVector with parameterLabels
     * @return arma::rowvec gradients
     */
    virtual arma::rowvec gradients(arma::rowvec parameterValues,
                                   stringVector parameterLabels) = 0;
  };

}
#endif