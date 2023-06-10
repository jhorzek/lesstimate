#ifndef FITRESULTS_H
#define FITRESULTS_H
#include "common_headers.h"

namespace lessSEM
{

  /**
   *
   * @struct fitResults
   * @brief The fit results returned by the optimizers.
   * @var fit the final fit value (regularized fit)
   * @var fits a vector with all fits at the outer iteration
   * @var convergence was the outer breaking condition met?
   * @var parameterValues final parameter values
   * @var Hessian final Hessian approximation (optional)
   */
  struct fitResults
  {
    double fit;
    arma::rowvec fits;
    bool convergence;
    arma::rowvec parameterValues;
    arma::mat Hessian;
  };

}

#endif