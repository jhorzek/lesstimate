#ifndef GLMNET_MCP
#define GLMNET_MCP

#include "penalty.h"

// IMPORTANT: MCP for glmnet is currently not very stable. We recommend
// using ista instead!

namespace lessSEM
{

class tuningParametersMcpGlmnet
{
public:
  arma::rowvec weights;
  double lambda;
  double theta;
};

class penaltyMcpGlmnet : public penalty<tuningParametersMcpGlmnet>
{
public:
  double getValue(const arma::rowvec &parameterValues,
                  const Rcpp::StringVector &parameterLabels,
                  const tuningParametersMcpGlmnet &tuningParameters)
  override
  {
    
    double penalty = 0.0;
    double lambda_i;
    double theta = tuningParameters.theta;
    double absPar;
    
    for (unsigned int p = 0; p < parameterValues.n_elem; p++)
    {
      
      if (tuningParameters.weights.at(p) == 0)
        continue;
      
      lambda_i = tuningParameters.lambda *
        tuningParameters.weights.at(p);
      
      absPar = std::abs(parameterValues.at(p));
      
      if(absPar <= (lambda_i * theta)){
        penalty += lambda_i * absPar - std::pow( absPar, 2 ) / (2.0 * theta);
        
      }else if(absPar > (lambda_i * theta)){
        
        penalty += theta * std::pow(lambda_i, 2) / 2.0;
        
      }else{
        Rcpp::stop("Error while evaluating mcp");
      }
      
    }
    
    return penalty;
  }
  
  // subproblemValue
  //
  // glmnet uses a combination of inner and outer iterations. Within the inner iteration, a
  // subproblem is solved for a single parameter. The mcp penalty is non-convex which
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
      
      double probe = std::abs(parameterValue_j + d_j + z);
      
      if(probe <= theta*lambda)
        return(
          base + lambda*probe - (probe * probe) / (2.0*theta)
        );
      
      return(
        base + theta*lambda*lambda/2.0
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
  // @param weight weight given to the penalty of this parameter
  double getZ(
      unsigned int whichPar,
      const arma::rowvec &parameters_kMinus1,
      const arma::rowvec &gradient,
      const arma::rowvec &stepDirection,
      const arma::mat &Hessian,
      const tuningParametersMcpGlmnet &tuningParameters)
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
    
    // Forming the second derivative of the functions below reveals an
    // issue:
    // d/dz (g_j + hessianXdirection_j + z * H_jj + lambda - (paramterValue_j + d_j + z) /(theta)) =
    // H_jj - (1/theta).
    // Note that the points we are deriving below are only a minima if H_jj - (1/theta) > 0.
    // Otherwise, they are maxima! Therefore, we also check the value H_jj - (1/theta):
    if(H_jj - 1.0/theta <= 0)
      Rcpp::warning("One of the subproblems for theta = %i and lambda = %j  is not positive definite. Using a small hack... This may work or may fail. We recommend using method = 'ista' for mcp.", theta, lambda);
    
    if(H_jj - (1/theta) <= 0){
      // We will make the function positive definite by replacing the Hessian approximation:
      H_jj += (1/theta) + .001;
    }
    
    // The problem we want to solve here is given by:
    // Find z such that
    // g_j + hessianXdirection_j + z * H_jj + d/dz p(paramterValue_j + d_j + z) = 0,
    // where 
    // p(paramterValue_j + d_j + z) = lambda * |paramterValue_j + d_j + z| - (paramterValue_j + d_j + z)^2 /(2*theta) if |paramterValue_j + d_j + z| <= lambda*theta
    // theta*lambda^2 / 2 otherwise
    
    // Mcp is non-convex, but has convex regions. We test 
    // all of these regions to check for the global minimum
    double z[3];
    double fitValue[3];
    
    // Case 1: |parameterValue_j + d_j + z| <= lambda*theta 
    // non-smooth penalty
    // p(parameterValue_j + d_j + z) = lambda * |parameterValue_j + d_j + z| - (parameterValue_j + d_j + z)^2 / (2*theta)
    
    // Assume that: parameterValue_j + d_j + z > 0 -> z > -(parameterValue_j + d_j)
    // In this case, the derivative of |parameterValue_j + d_j + z| wrt z is 1. It follows:
    // g_j + hessianXdirection_j + z * H_jj + d/dz p(paramterValue_j + d_j + z) = 
    // g_j + hessianXdirection_j + z * H_jj + lambda - (paramterValue_j + d_j + z) /(theta) = 0
    double z_1 = std::max(
      -(parameterValue_j + d_j), // note: this sets the parameter to zero
      (-hessianXdirection_j*theta + d_j - g_j*theta - theta*lambda + parameterValue_j)/(H_jj*theta - 1.0)
    );
    // additionally, parameterValue_j + d_j + z must be <= lambda*theta -> z <= lambda*theta - (parameterValue_j + d_j)
    if(parameterValue_j + d_j + z_1 <= lambda*theta){
      z[0] = z_1;
    }else{
      z[0] = lambda*theta - (parameterValue_j + d_j);
    }
    
    // Assume that: parameterValue_j + d_j + z < 0 -> z < -(parameterValue_j + d_j)
    // In this case, the derivative of |parameterValue_j + d_j + z| wrt z is -1. It follows:
    // g_j + hessianXdirection_j + z * H_jj + d/dz p(paramterValue_j + d_j + z) = 
    // g_j + hessianXdirection_j + z * H_jj - lambda - (paramterValue_j + d_j + z) /(theta) = 0
    double z_2 = std::min(
      -(parameterValue_j + d_j), // note: this sets the parameter to zero
      (-hessianXdirection_j*theta + d_j - g_j*theta + theta*lambda + parameterValue_j)/(H_jj*theta - 1.0)
    );
    // additionally, parameterValue_j + d_j + z must be >= -lambda*theta -> z >= -lambda*theta - (parameterValue_j + d_j)
    if(parameterValue_j + d_j + z_2 >= -lambda*theta){
      z[1] = z_2;
    }else{
      z[1] = -lambda*theta - (parameterValue_j + d_j);
    }
    
    // Case 2: |parameterValue_j + d_j + z| > lambda*theta 
    
    // p(parameterValue_j + d_j + z) = theta*lambda^2 / 2 
    // It follows:
    // g_j + hessianXdirection_j + z * H_jj + d/dz p(paramterValue_j + d_j + z) = 
    // g_j + hessianXdirection_j + z * H_jj = 0
    double z_3 = -(g_j + hessianXdirection_j)/H_jj;
    
    // We also have to make sure that our parameterValue_j + d_j + z_3 is outside
    // of |lambda*theta|:
    
    if(parameterValue_j + d_j + z_3 < 0){
      // Case 2.1
      // parameterValue_j + d_j + z < 0 and
      // parameterValue_j + d_j + z < -lambda*theta -> z < -lambda*theta - (parameterValue_j + d_j)
      if(parameterValue_j + d_j + z_3 <= -lambda*theta){
        z[2] = z_3;
      }else{
        z[2] = -lambda*theta - (parameterValue_j + d_j);
      }
    }else{
      // Case 2.2
      // parameterValue_j + d_j + z > 0 and
      // parameterValue_j + d_j + z > lambda*theta -> z > lambda*theta - (parameterValue_j + d_j)
      if(parameterValue_j + d_j + z_3 >= lambda*theta){
        z[2] = z_3;
      }else{
        z[2] = lambda*theta - (parameterValue_j + d_j);
      }
    }
    
    // For completeness sake, we also check the boundaries explicitly
    // z[2] = lambda*theta - (parameterValue_j + d_j);
    // z[3] = -lambda*theta - (parameterValue_j + d_j);  
    
    // compute fit value
    int whichmin = -1;
    for (unsigned int i = 0; i < 3; i++)
    {
      
      if(!arma::is_finite(z[i]))
        continue;
      
      fitValue[i] = this->subproblemValue(
        parameterValue_j,
        z[i],
         g_j,
         d_j,
         hessianXdirection_j,
         H_jj,
         lambda,
         theta);
      
      if (whichmin == -1){
        whichmin = i;
      }else {
        if (fitValue[i] < fitValue[whichmin])
          whichmin = i;
      }
    }
    if(whichmin == -1){
      Rcpp::stop("Found no minimum");
    }
    
    return (z[whichmin]);
  }
  
  arma::rowvec
  getSubgradients(const arma::rowvec &parameterValues,
                  const arma::rowvec &gradients,
                  const tuningParametersMcpGlmnet &tuningParameters)
  {
    Rcpp::stop("Subgradients not yet implemented for mcp penalty. Use different convergence criterion.");
  }
};
}

#endif