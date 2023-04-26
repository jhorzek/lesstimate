#ifndef LSP_GLMNET_H
#define LSP_GLMNET_H
#include "common_headers.h"

#include "penalty.h"

namespace lessSEM
{

class tuningParametersLspGlmnet
{
public:
  arma::rowvec weights;
  double lambda;
  double theta;
};

// the glmnet penalty allows for vectors of alpha and lambda
class penaltyLSPGlmnet : public penalty<tuningParametersLspGlmnet>
{
public:
  double getValue(const arma::rowvec &parameterValues,
                  const Rcpp::StringVector &parameterLabels,
                  const tuningParametersLspGlmnet &tuningParameters)
  override
  {
    
    double penalty = 0.0;
    
    for (unsigned int p = 0; p < parameterValues.n_elem; p++)
    {
      
      if (tuningParameters.weights.at(p) == 0)
        continue;
      
      double lambda = tuningParameters.weights.at(p) * tuningParameters.lambda;
      double theta = tuningParameters.theta;
      
      penalty += lambda * std::log(1.0 + std::abs(parameterValues.at(p)) / theta);
      
    }
    
    return penalty;
  }
  
  // subproblemValue
  //
  // glmnet uses a combination of inner and outer iterations. Within the inner iteration, a
  // subproblem is solved for a single parameter. The lsp penalty is non-convex which
  // means that there may be local minima in the subproblem. However, because the function is
  // convex within regions, we can find the minimum within each region and then compare the results
  // to find the global minimum. To this end, we need the function value of the subproblem. This
  // is computed here.
  // @param parameterValue_j parameter value from the outer iteration for parameter j
  // @param z update for parameter j in current inner iteration
  // @param g_j gradient value from the outer iteration for parameter j
  // @param d_j direction value from the inner iteration for parameter j
  // @param hessianXdirection_j product of hessian and direction parameter value from the outer iteration for parameter j
  // @param H_jj row j, col j of Hessian matrix
  // @param lambda tuning parameter lambda
  // @param theta tuning parameter theta
  // @return fit value (double)
  double subproblemValue(
      const double parameterValue_j,
      const double z,
      const double g_j,
      const double d_j,
      const double hessianXdirection_j,
      const double H_jj,
      const double lambda,
      const double theta)
  {
    double base = z * g_j +
      z * hessianXdirection_j +
      .5 * (z * z) * H_jj;
      
      return(
        base + lambda * std::log(1.0 + std::abs(parameterValue_j + d_j + z) / theta)
      );
  }
  
  // getZ
  //
  // computes the step direction for a single parameter j in the inner
  // iterations of the lasso penalty.
  // @param d_j gradient of the smooth part for parameter j
  // @param H_jj Hessian in row and column j
  // @param hessianXdirection_j element j from product of Hessian and direction
  // @param alpha tuning parameter alpha
  // @param lambda tuning parameter lambda
  // @param weight weight given to the penalty of this parameter (relevant in adaptive lasso)
  double getZ(
      unsigned int whichPar,
      const arma::rowvec &parameters_kMinus1,
      const arma::rowvec &gradient,
      const arma::rowvec &stepDirection,
      const arma::mat &Hessian,
      const tuningParametersLspGlmnet &tuningParameters)
  {
    double lambda = tuningParameters.weights.at(whichPar) * tuningParameters.lambda;
    double theta = tuningParameters.theta;
    
    double parameterValue_j = arma::as_scalar(parameters_kMinus1.col(whichPar));
    
    // compute derivative elements:
    double d_j = arma::as_scalar(stepDirection.col(whichPar));
    arma::colvec hessianXdirection = Hessian * arma::trans(stepDirection);
    double hessianXdirection_j = arma::as_scalar(hessianXdirection.row(whichPar));
    double H_jj = arma::as_scalar(Hessian.row(whichPar).col(whichPar));
    double g_j = arma::as_scalar(gradient.col(whichPar));
    
    if (tuningParameters.weights.at(whichPar) == 0)
    {
      // No regularization
      return (-(g_j + hessianXdirection_j) / H_jj);
    }
    
    // The lsp penalty is non-convex and may have multiple minima.
    // However, there are parts of the function that are convex. We
    // can therefore check these parts for their respective minima and
    // then check which is the overall minimum. The implementation below is 
    // a bit messy, so we will go through all the steps along the way.
    double z[5]; // we will test 5 different values
    double fitValue[5];
    
    // We want to minimize the function 
    // z * g_j + z * hessianXdirection_j + .5 * (z * z) * H_jj + lambda * log(1.0 + |parameterValue_j + d_j + z| / theta)
    // To this end, we first create the derivative of the function wrt z. This gives:
    // g_j + hessianXdirection_j + H_jj * z + d/dz log(1.0 + |parameterValue_j + d_j + z| / theta)
    // Note that the last part is not differentiable. However, we can use some tricks.
    // In general, 
    // d/dz log(1.0 + |parameterValue_j + d_j + z| / theta) = 
    //    (1/(1.0 + |parameterValue_j + d_j + z| / theta)) * (d/dz |parameterValue_j + d_j + z|/theta)
    
    // First, assume that parameterValue_j + d_j + z > 0. It
    // follows that the derivative of |parameterValue_j + d_j + z| with 
    // respect  to z is 1. Furthermore, we can replace |parameterValue_j + d_j + z|
    // with parameterValue_j + d_j + z. Doing so allows for rewriting the equation
    // g_j + hessianXdirection_j + H_jj * z + d/dz log(1.0 + |parameterValue_j + d_j + z| / theta)  = 0
    // as a midnight formula. The solutions for this are given below:
    double v1 = g_j + hessianXdirection_j + H_jj*theta + H_jj * parameterValue_j + H_jj*d_j;
    double v2 = H_jj;
    double v3 = -g_j*theta - g_j * parameterValue_j - g_j*d_j -
      hessianXdirection_j * theta - hessianXdirection_j *parameterValue_j - 
      hessianXdirection_j*d_j - lambda;
    
    if(v1*v1 + 4*v2*v3 >= 0){
      z[0] = -(v1 + std::sqrt(v1*v1 + 4*v2*v3))/(2*v2);
      z[1] = -(v1 - std::sqrt(v1*v1 + 4*v2*v3))/(2*v2);
    }else{
      z[0] = arma::datum::nan;
      z[1] = arma::datum::nan;
    }
    
    // Second, assume that parameterValue_j + d_j + z < 0. It
    // follows that the derivative of |parameterValue_j + d_j + z| with 
    // respect  to z is -1. Furthermore, we can replace |parameterValue_j + d_j + z|
    // with -(parameterValue_j + d_j + z). Doing so allows for rewriting the equation
    // g_j + hessianXdirection_j + H_jj * z + d/dz log(1.0 + |parameterValue_j + d_j + z| / theta)  = 0
    // as a midnight formula. The solutions for this are given below:
    
    double m1 = -g_j - hessianXdirection_j + H_jj*theta - H_jj * parameterValue_j - H_jj*d_j;
    double m2 = H_jj;
    double m3 = -g_j*theta + g_j * parameterValue_j + g_j*d_j -
      hessianXdirection_j * theta + hessianXdirection_j *parameterValue_j + 
      hessianXdirection_j*d_j + lambda;
    
    if(m1*m1 - 4*m2*m3 >= 0){
      z[2] = (m1 + std::sqrt(m1*m1 - 4*m2*m3))/(2*m2);
      z[3] = (m1 - std::sqrt(m1*m1 - 4*m2*m3))/(2*m2);
    }else{
      z[2] = arma::datum::nan;
      z[3] = arma::datum::nan;
    }
    
    // Finally, assume that parameterValue_j + d_j + z = 0.
    z[4] = -(parameterValue_j + d_j);
    
    // We now compute the fit for all five of our possible solutions
    // and select the one which results in the lowest fit value.
    int whichmin = -1;
    for (unsigned int i = 0; i < 5; i++)
    {
      
      if(!arma::is_finite(z[i])){
        continue;
        }
      
      fitValue[i] = this->subproblemValue(
        parameterValue_j,
        z[i],
         g_j,
         d_j,
         hessianXdirection_j,
         H_jj,
         lambda,
         theta);
      
      if(whichmin == -1){
        whichmin = i;
      } else {
        if (fitValue[i] < fitValue[whichmin])
          whichmin = i;
      }
    }
    
    if(whichmin == -1)
      Rcpp::stop("Could not find a minimum.");
    
    return (z[whichmin]);
  }
  
  arma::rowvec getSubgradients(const arma::rowvec &parameterValues,
                               const arma::rowvec &gradients,
                               const tuningParametersLspGlmnet &tuningParameters)
  {
    Rcpp::stop("Subgradients not yet implemented for lsp penalty. Use different convergence criterion.");
  }
};

}

#endif