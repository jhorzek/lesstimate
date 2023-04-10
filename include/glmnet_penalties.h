#ifndef MIXEDPENALTYGLMNET_H
#define MIXEDPENALTYGLMNET_H
#include <RcppArmadillo.h>
#include "penalty_type.h"
#include "cappedL1.h"
#include "lasso.h"
#include "lsp.h"
#include "mcp.h"
#include "scad.h"

namespace lessSEM{

// to find the best z value, we have to check which one results in the 
// lowest fit function value. This will be checked in the following function:
inline double getPartialFitValue(
    const double parameterValue_j,
    const double z,
    const double g_j,
    const double d_j,
    const double hessianXdirection_j,
    const double H_jj,
    const penaltyType penalty,
    const double weight,
    const double lambda,
    const double theta,
    const double alpha
){
  
  // fit without penalty:
  double base =  z *g_j +
    z*hessianXdirection_j +
    .5*(z*z)*H_jj;
    
  // add penalty
    if(penalty == lasso){
      return(base + 
             lambda*alpha*weight*std::abs(parameterValue_j + d_j + z) - 
             lambda*alpha*weight*std::abs(parameterValue_j + d_j)
      );
    }
    
    if(penalty == scad){
      double probe = std::abs(parameterValue_j + d_j + z);
      
      if(probe <= theta)
        return(base + lambda*probe);
      
      if(lambda < probe && probe <= lambda*theta)
        return(base + (-probe*probe + 2.0*theta*lambda*probe - lambda*lambda)/(2.0*(theta-1.0)));
      
      if(probe >= lambda*theta)
        return(base + (theta + 1.0)*lambda*lambda/2.0);
      
    }
    
    Rcpp::stop("Penalty not yet implemented");
}


inline double getZs(
    const double parameterValue_j, // previous value of parameter j
    const double g_j, // gradient of parameter j 
    const double H_jj, // Hessian in row and column j
    const double d_j, // direction of parameter j
    const double hessianXdirection_j, // H*d'
    // penalty
    const penaltyType penalty,
    // tuning parameters:
    const double weight,
    const double lambda,
    const double theta,
    const double alpha
){
  
  if(penalty == lasso){
    double tuning = weight * lambda * alpha;
    double probe = parameterValue_j + d_j - (g_j + hessianXdirection_j)/H_jj;
    
    if(probe - tuning/H_jj > 0)
      return(-(g_j + hessianXdirection_j + tuning)/H_jj);
    
    if(probe + tuning/H_jj < 0)
      return(-(g_j + hessianXdirection_j - tuning)/H_jj);
    
    return(-parameterValue_j - d_j);
    
  }else if(penalty == scad){
    
    // We have to check multiple cases for their minima
    double z[3];
    double fitValue[3];
    
    // Case 1: lasso
    double probe1 = parameterValue_j + d_j + (-g_j + hessianXdirection_j + lambda)/theta;
    double probe2 = parameterValue_j + d_j + (-g_j + hessianXdirection_j - lambda)/theta;
    
    if((probe1 > 0) & (std::abs(probe1) <= theta)){
      z[0] = (-g_j + hessianXdirection_j + lambda)/theta;
    }else if((probe2 < 0) & (std::abs(probe2) <= theta)){
      z[0] = (-g_j + hessianXdirection_j - lambda)/theta;
    }else{
      z[0] = -parameterValue_j - d_j;
    }
    
    // Case 2:
    probe1 = (-2.0*hessianXdirection_j*(theta-1) + 
      lambda*(-2.0*d_j - 2.0*theta + lambda -2.0*parameterValue_j) -
      2.0*g_j*(theta-1))/(2.0*(H_jj*(theta-1.0)+lambda));
    probe2 = (-2.0*hessianXdirection_j*(theta-1) + 
      lambda*(-2.0*d_j + 2.0*theta + lambda - 2.0*parameterValue_j) -
      2.0*g_j*(theta-1))/(2.0*(H_jj*(theta-1.0)+lambda));
    
    if((parameterValue_j + d_j + probe1  > lambda)&
       (lambda*theta >= parameterValue_j + d_j + probe1)){
      z[1] = probe1;
    }else if((parameterValue_j + d_j + probe1 < lambda)&
      (lambda*theta >= std::abs(parameterValue_j + d_j + probe1))){
      z[1] = probe2;
    }else{
      z[1] = arma::datum::nan;
    }
    
    // Case 3:
    probe1 = -(g_j + hessianXdirection_j)/H_jj;
    
    if(std::abs(parameterValue_j + d_j + probe1) >= theta*lambda){
      z[2] = probe1;
    }else{
      z[2] = arma::datum::nan;
    }
    
    // compute fit value
    int whichmin = 0;
    bool changed = false;
    for(unsigned int i = 0; i < 3; i++){
      if(!arma::is_finite(z[i])){
        continue;
      }else{
        changed = true;
      }
      
      fitValue[i] = getPartialFitValue(
        parameterValue_j,
        z[i],
         g_j,
         d_j,
         hessianXdirection_j,
         H_jj,
         // penalty
         penalty,
         // tuning parameters:
         weight,
         lambda,
         theta,
         alpha
      );
      
      if(i > 0){
        if(fitValue[i] < fitValue[whichmin])
          whichmin = i;
      }
    }
    if(!changed)
      Rcpp::stop("Error: none of the elements was valid.");
    
    return(z[whichmin]);
  }
  
  Rcpp::stop("Did not find penalty function.");

}

}

#endif
