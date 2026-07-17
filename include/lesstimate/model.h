#ifndef MODEL_H
#define MODEL_H

#include "common_headers.h"

namespace lessSEM
{
  /**
   * @brief model is the base class used in every optimizer implemented in lesstimate.
   * The user specified model should inherit from the model class and must implement
   * the two methods defined therein
   */
  class model
  {
  public:

  virtual ~model() = default;

    /**
     * @brief fit method  with arguments parameterValues (arma::rowvec) and parameterLabels (stringVector; see common_headers.h)
     * specifying the parameter values and the labels of the paramters. The function should return the fit value (double).
     *
     * @param parameterValues numericVector with parameter values
     * @param parameterLabels stringVector with parameterLabels
     * @return double
     */
    virtual double fit(const arma::rowvec& parameterValues,
                       const stringVector& parameterLabels) = 0;

    /**
     * @brief gradients method with arguments parameterValues(arma::rowvec) and parameterLabels(stringVector; see common_headers.h) * specifying the parameter values and the labels of the paramters. The function should return the gradients(arma::rowvec).
     * By default, a central gradient approximation with step size 1e-5 is used
     *
     * @param parameterValues numericVector with parameter values
     * @param parameterLabels stringVector with parameterLabels
     * @return arma::rowvec gradients
     */
    virtual arma::rowvec gradients(const arma::rowvec& parameterValues,
                                   const stringVector& parameterLabels)
    {
      // The default central-difference implementation mutates the parameter
      // vector to apply +/- stepSize. With a const reference interface, take a
      // local copy here; the caller's value is left untouched.
      arma::rowvec perturbedParameters = parameterValues;
      arma::rowvec gradients(perturbedParameters.n_elem);
      gradients.fill(arma::fill::zeros);
      // define stepSize used in numerically approximated gradients:
      double stepSize = 1e-5;
      
      for (unsigned int i = 0; i < perturbedParameters.n_elem; i++)
      {

        // step forward
        perturbedParameters(i) += stepSize;
        gradients(i) = fit(perturbedParameters,
                           parameterLabels);

        // step backward
        perturbedParameters(i) -= 2.0 * stepSize;
        gradients(i) -= fit(perturbedParameters,
                            parameterLabels);
        // reset
        perturbedParameters(i) += stepSize;

        // compute gradient
        gradients(i) /= 2.0 * stepSize;
      }
      return (gradients);
    }
  };

}
#endif