#ifndef PENALTY_H
#define PENALTY_H
#include "common_headers.h"

namespace lessSEM{

/**
 * @brief penalty class
 * 
 * @tparam T type of the tuning parameters used by the penalty
 */
template<class T>
class penalty{
public:
  /**
   * @brief return the value of the penalty function
   * 
   * @param parameterValues current parameter values
   * @param parameterLabels parameter labels
   * @param tuningParameters tuning parameters of the penalty function
   * @return double 
   */
  virtual double getValue(const arma::rowvec& parameterValues,
                          const stringVector& parameterLabels,
                          const T& tuningParameters);
};
}

#endif