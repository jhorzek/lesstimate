#ifndef PROXIMALOPERATOR_H
#define PROXIMALOPERATOR_H
#include "common_headers.h"

namespace lessSEM{
  /**
   * @brief proximal operator of the ista optimizer
   * 
   * @tparam T tuning parameter class
   */
template<class T>
class proximalOperator{
public:
/**
 * @brief return the parameters after updating with the proximal operator
 * 
 * @param parameterValues current parameter values
 * @param gradientValues current gradient values
 * @param parameterLabels parameter labels
 * @param L step length
 * @param tuningParameters tuning parameters of the penalty function 
 * @return arma::rowvec updated parameters
 */
  virtual arma::rowvec getParameters(const arma::rowvec& parameterValues, 
                                            const arma::rowvec& gradientValues,
                                            const stringVector& parameterLabels,
                                            const double L,
                                            const T& tuningParameters) = 0;
};
}
#endif