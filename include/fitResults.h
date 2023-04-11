#ifndef FITRESULTS_H
#define FITRESULTS_H

namespace lessSEM{

// fitResults
//
// The fit results returned by the optimizers.
//
// @param fit the final fit value (regularized fit)
// @param fits a vector with all fits at the outer iteration
// @param convergence was the outer breaking condition met?
// @param parameterValues final parameter values
// @param Hessian final Hessian approximation (optional)
struct fitResults{
  double fit;
  arma::rowvec fits;
  bool convergence;
  arma::rowvec parameterValues;
  arma::mat Hessian;
};

}

#endif