#ifndef SMOOTHPENALTY_H
#define SMOOTHPENALTY_H
#include "common_headers.h"

namespace lessSEM
{
  /**
   * @brief smooth penalty function
   *
   * @tparam T tuning parameters
   */
  template <class T>
  class smoothPenalty
  {
  public:

  virtual ~smoothPenalty() = default;
    /**
     * @brief Get the value of the penalty function
     *
     * @param parameterValues current parameter values
     * @param parameterLabels names of the parameters
     * @param tuningParameters values of the tuning parmameters
     * @return double
     */
    virtual double getValue(const arma::rowvec &parameterValues,
                            const stringVector &parameterLabels,
                            const T &tuningParameters) = 0;
    /**
     * @brief returns gradients of the penalty function
     *
     * @param parameterValues current parameter values
     * @param parameterLabels names of the parameters
     * @param tuningParameters values of the tuning parmameters
     * @return arma::rowvec
     */
    virtual arma::rowvec getGradients(const arma::rowvec &parameterValues,
                                      const stringVector &parameterLabels,
                                      const T &tuningParameters) = 0;
  };

  // define some smooth penalties:
  /**
   * @brief no smooth penalty
   * 
   * @tparam T tuning parameters
   */
  template <class T>
  class noSmoothPenalty : public smoothPenalty<T>
  {
  public:
    /**
     * @brief Get the value of the penalty function. Returns zero
     *
     * @param parameterValues current parameter values
     * @param parameterLabels names of the parameters
     * @param tuningParameters values of the tuning parmameters
     * @return double
     */
    double getValue(const arma::rowvec &parameterValues,
                    const stringVector &parameterLabels,
                    const T &tuningParameters) override
    {
      static_cast<void>(parameterValues); // is unused
      static_cast<void>(parameterLabels); // is unused
      static_cast<void>(tuningParameters); // is unused
      return (0.0);
    };
        /**
     * @brief returns gradients of the penalty function. Returns a vector of zeros
     *
     * @param parameterValues current parameter values
     * @param parameterLabels names of the parameters
     * @param tuningParameters values of the tuning parmameters
     * @return arma::rowvec
     */
    arma::rowvec getGradients(const arma::rowvec &parameterValues,
                              const stringVector &parameterLabels,
                              const T &tuningParameters) override
    {
      static_cast<void>(parameterLabels); // is unused
      static_cast<void>(tuningParameters); // is unused
      arma::rowvec gradients(parameterValues.n_elem);
      gradients.fill(0.0);
      return (gradients);
    };
  };

/**
 * @brief tuning parameters of the elastic net
 * 
 */
  struct tuningParametersSmoothElasticNet
  {
    double lambda; ///> lambda tuning parameter
    double alpha; //> alpha tuning parameter
    double epsilon; ///> to make function smooth
    arma::rowvec weights; ///> parameter-specific weights for penalties
  };

/**
 * @brief smoothed elastic net penalty function 
 * 
 */
  class smoothElasticNet : public smoothPenalty<tuningParametersSmoothElasticNet>
  {

  public:
      /**
     * @brief Get the value of the penalty function
     *
     * @param parameterValues current parameter values
     * @param parameterLabels names of the parameters
     * @param tuningParameters values of the tuning parmameters
     * @return double
     */
    double getValue(const arma::rowvec &parameterValues,
                    const stringVector &parameterLabels,
                    const tuningParametersSmoothElasticNet &tuningParameters)
        override
    {

      static_cast<void>(parameterLabels); // is unused, but necessary for the interface
      double penalty = 0.0;
      double lambda_i;

      for (unsigned int p = 0; p < parameterValues.n_elem; p++)
      {
        // lasso part:
        lambda_i = tuningParameters.alpha *
                   tuningParameters.lambda *
                   tuningParameters.weights.at(p);

        penalty += lambda_i *
                   std::sqrt(std::pow(parameterValues.at(p), 2) + tuningParameters.epsilon);

        // ridge part
        lambda_i = (1.0 - tuningParameters.alpha) *
                   tuningParameters.lambda *
                   tuningParameters.weights.at(p);
        penalty += lambda_i *
                   std::pow(parameterValues.at(p), 2);
      }

      return penalty;
    }

    /**
     * @brief Get the gradients of the penalty function
     *
     * @param parameterValues current parameter values
     * @param parameterLabels names of the parameters
     * @param tuningParameters values of the tuning parmameters
     * @return arma::rowvec
     */
    arma::rowvec getGradients(const arma::rowvec &parameterValues,
                              const stringVector &parameterLabels,
                              const tuningParametersSmoothElasticNet &tuningParameters) override
    {

      static_cast<void>(parameterLabels); // is unused, but necessary for the interface
      arma::rowvec gradients(parameterValues.n_elem);
      gradients.fill(0.0);
      double lambda_i;

      for (unsigned int p = 0; p < parameterValues.n_elem; p++)
      {

        // if not regularized: nothing to do here
        if (tuningParameters.weights.at(p) == 0)
          continue;

        // lasso part:
        lambda_i = tuningParameters.alpha *
                   tuningParameters.lambda *
                   tuningParameters.weights.at(p);

        gradients.at(p) += lambda_i * parameterValues.at(p) *
                           (1.0 / std::sqrt(std::pow(parameterValues.at(p), 2) +
                                            tuningParameters.epsilon));

        // ridge part
        lambda_i = (1.0 - tuningParameters.alpha) *
                   tuningParameters.lambda *
                   tuningParameters.weights.at(p);
        gradients.at(p) += lambda_i *
                           2.0 *
                           parameterValues.at(p);

      } // end for parameter

      return (gradients);
    }
  };

} // end namespace
#endif