#ifndef PENALTY_H
#define PENALTY_H
#include "common_headers.h"

namespace lessSEM{

template<class T>
class penalty{
public:
  
  virtual double getValue(const arma::rowvec& parameterValues,
                          const Rcpp::StringVector& parameterLabels,
                          const T& tuningParameters);
};
}

#endif