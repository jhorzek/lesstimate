#ifndef PROXIMALOPERATOR_H
#define PROXIMALOPERATOR_H
#include "common_headers.h"

namespace lessSEM{
template<class T>
class proximalOperator{
public:
  virtual arma::rowvec getParameters(const arma::rowvec& parameterValues, 
                                            const arma::rowvec& gradientValues,
                                            const stringVector& parameterLabels,
                                            const double L,
                                            const T& tuningParameters) = 0;
};
}
#endif