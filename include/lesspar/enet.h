#ifndef ENET_H
#define ENET_H
#include "common_headers.h"

// The elastic net is a combination of ridge and lasso
// In this file, only the tuning parameters required by lasso
// and ridge are defined.
namespace lessSEM
{
  /**
   * @brief tuning parameters of the elastic net penalty
   *
   */
  class tuningParametersEnet
  {
  public:
    double lambda;        ///> lambda value >= 0
    double alpha;         ///> alpha value of the elastic net (relative importance of ridge and
    arma::rowvec weights; ///> provide parameter-specific weights (e.g., for adaptive lasso)
  };

  // for glmnet, we will allow for different alphas and lambdas to combine penalties
  /**
   * @brief tuning parameters of the elastic net penalty
   *
   */
  class tuningParametersEnetGlmnet
  {
  public:
    arma::rowvec lambda;  ///> parameter-specific lambda value >= 0
    arma::rowvec alpha;   ///> parameter-specific alpha value of the elastic net (relative importance of ridge and
    arma::rowvec weights; ///> provide parameter-specific weights (e.g., for adaptive lasso)
  };
}

#endif