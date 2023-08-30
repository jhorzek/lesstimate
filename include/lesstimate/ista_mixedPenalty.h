#ifndef MIXEDPENALTY_H
#define MIXEDPENALTY_H
#include <memory>
#include "common_headers.h"

#include "penalty_type.h"
#include "proximalOperator.h"
#include "penalty.h"
#include "ista_cappedL1.h"
#include "ista_lasso.h"
#include "ista_lsp.h"
#include "ista_mcp.h"
#include "ista_scad.h"

// The proximal operator for this penalty function has been developed by
// Gong, P., Zhang, C., Lu, Z., Huang, J., & Ye, J. (2013).
// A general iterative shrinkage and thresholding algorithm for non-convex
// regularized optimization problems. Proceedings of the 30th International
// Conference on Machine Learning, 28(2)(2), 37–45.

// The implementation deviated from that of Gong et al. (2013). We
// could not get the procedure outlined by Gong et al. (2013) to return the
// correct results. In the following, we will use our own derivation, but
// it is most likely equivalent to that of Gong et al. (2013) and we
// just failed to implement their approach correctly. If you find our
// mistake, feel free to improve upon the following implementation.

namespace lessSEM
{
  /**
   * @brief tuning parameters for the mixed penalty using ista
   *
   */
  class tuningParametersMixedPenalty
  {
  public:
    arma::rowvec lambda;         ///> paramter-specific lambda value
    arma::rowvec theta;          ///> paramter-specific theta value
    arma::rowvec alpha;          ///> paramter-specific alpha value
    arma::rowvec weights;        ///> paramter-specific weights
    std::vector<penaltyType> pt; ///> penalty type
  };

/**
 * @brief base class for proximal operator for the mixed penalty function
 *
 */
class proximalOperatorMixedBase {
public:
  virtual arma::rowvec getParameters(const arma::rowvec &parameterValues,
                             const arma::rowvec &gradientValues,
                             const stringVector &parameterLabels,
                             const double L,
                             const tuningParametersMixedPenalty &tuningParameters) = 0;
};

class proximalOperatorMixedNone: public proximalOperatorMixedBase{
public:
  arma::rowvec getParameters(const arma::rowvec &parameterValues,
                             const arma::rowvec &gradientValues,
                             const stringVector &parameterLabels,
                             const double L,
                             const tuningParametersMixedPenalty &tuningParameters) override{
                              static_cast<void>(parameterLabels); // is unused, but necessary for the interface to be consistent
                              static_cast<void>(tuningParameters); // is unused, but necessary for the interface to be consistent
                               arma::rowvec u_k = parameterValues - gradientValues / L;
                               return(u_k);
                             }
};


class proximalOperatorMixedCappedL1: public proximalOperatorMixedBase{
public:
  arma::rowvec getParameters(const arma::rowvec &parameterValues,
                             const arma::rowvec &gradientValues,
                             const stringVector &parameterLabels,
                             const double L,
                             const tuningParametersMixedPenalty &tuningParameters) override{
                               
                               tp.alpha = tuningParameters.alpha(0);
                               tp.lambda = tuningParameters.lambda(0);
                               tp.theta = tuningParameters.theta(0);
                               tp.weights = tuningParameters.weights(0);
                               
                               return(
                                 proxOp.getParameters(
                                   parameterValues,
                                   gradientValues,
                                   parameterLabels,
                                   L,
                                   tp
                                 )
                               );
                             }
private: 
  tuningParametersCappedL1 tp;
  proximalOperatorCappedL1 proxOp;
};

class proximalOperatorMixedLasso: public proximalOperatorMixedBase{
public:
  arma::rowvec getParameters(const arma::rowvec &parameterValues,
                             const arma::rowvec &gradientValues,
                             const stringVector &parameterLabels,
                             const double L,
                             const tuningParametersMixedPenalty &tuningParameters) override{
                               
                               tp.alpha = tuningParameters.alpha(0);
                               tp.lambda = tuningParameters.lambda(0);
                               tp.weights = tuningParameters.weights(0);
                               
                               return(
                               proxOp.getParameters(
                                 parameterValues,
                                 gradientValues,
                                 parameterLabels,
                                 L,
                                 tp
                               )
                               );
                             }
private: 
  tuningParametersEnet tp;
  proximalOperatorLasso proxOp;
};

class proximalOperatorMixedLsp: public proximalOperatorMixedBase{
public:
  arma::rowvec getParameters(const arma::rowvec &parameterValues,
                             const arma::rowvec &gradientValues,
                             const stringVector &parameterLabels,
                             const double L,
                             const tuningParametersMixedPenalty &tuningParameters) override{
                               
                               tp.lambda = tuningParameters.lambda(0);
                               tp.theta = tuningParameters.theta(0);
                               tp.weights = tuningParameters.weights(0);
                               
                               return(
                                 proxOp.getParameters(
                                   parameterValues,
                                   gradientValues,
                                   parameterLabels,
                                   L,
                                   tp
                                 )
                               );
                             }
private: 
  tuningParametersLSP tp;
  proximalOperatorLSP proxOp;
};

class proximalOperatorMixedMcp: public proximalOperatorMixedBase{
public:
  arma::rowvec getParameters(const arma::rowvec &parameterValues,
                             const arma::rowvec &gradientValues,
                             const stringVector &parameterLabels,
                             const double L,
                             const tuningParametersMixedPenalty &tuningParameters) override{
                               
                               tp.lambda = tuningParameters.lambda(0);
                               tp.theta = tuningParameters.theta(0);
                               tp.weights = tuningParameters.weights(0);
                               
                               return(
                                 proxOp.getParameters(
                                   parameterValues,
                                   gradientValues,
                                   parameterLabels,
                                   L,
                                   tp
                                 )
                               );
                             }
private: 
  tuningParametersMcp tp;
  proximalOperatorMcp proxOp;
};


class proximalOperatorMixedScad: public proximalOperatorMixedBase{
public:
  arma::rowvec getParameters(const arma::rowvec &parameterValues,
                             const arma::rowvec &gradientValues,
                             const stringVector &parameterLabels,
                             const double L,
                             const tuningParametersMixedPenalty &tuningParameters) override{
                               
                               tp.lambda = tuningParameters.lambda(0);
                               tp.theta = tuningParameters.theta(0);
                               tp.weights = tuningParameters.weights(0);
                               
                               return(
                                 proxOp.getParameters(
                                   parameterValues,
                                   gradientValues,
                                   parameterLabels,
                                   L,
                                   tp
                                 )
                               );
                             }
private: 
  tuningParametersScad tp;
  proximalOperatorScad proxOp;
};

class proximalOperatorMixedPenalty: public proximalOperator<tuningParametersMixedPenalty> {
public:
  std::vector<std::unique_ptr<proximalOperatorMixedBase>> proxOps;
  
  arma::rowvec getParameters(
     const arma::rowvec &parameterValues,
     const arma::rowvec &gradientValues,
     const stringVector &parameterLabels,
     const double L,
     const tuningParametersMixedPenalty &tuningParameters) override {
        
       arma::rowvec parameterValue{0};
       arma::rowvec gradientValue{0};
       arma::rowvec parameters_kp1 = parameterValues;
       
       int it = 0;
       for(auto& proxOp: proxOps){
         tpSinglePenalty.alpha = tuningParameters.alpha(it);
         tpSinglePenalty.lambda = tuningParameters.lambda(it);
         tpSinglePenalty.theta = tuningParameters.theta(it);
         tpSinglePenalty.weights = tuningParameters.weights(it);
         
         parameterValue(0) = parameterValues(it);
         gradientValue(0) = gradientValues(it);
         
         parameters_kp1(it) = arma::as_scalar(proxOp->getParameters(
                                                        parameterValue,
                                                        gradientValue,
                                                        parameterLabels,
                                                        L,
                                                        tpSinglePenalty)
                                                );
         it++;
       }
       
       return(parameters_kp1);
                                       
     }
  
private:
  tuningParametersMixedPenalty tpSinglePenalty;
};

/**
 * @brief base class for mixed penalty
 *
 */
class penaltyMixedPenaltyBase
{
public:
  /**
   * @brief Get the value of the penalty function
   *
   * @param parameterValues current parameter values
   * @param parameterLabels names of the parameters
   * @param tuningParameters values of the tuning parmameters
   * @return double
   */
  virtual double getValue(const arma::rowvec &parameterValues,
                  const stringVector &parameterLabels,
                  const tuningParametersMixedPenalty &tuningParameters) = 0;
};

class penaltyMixedNone: public penaltyMixedPenaltyBase{
public:
  double getValue(const arma::rowvec &parameterValues,
                        const stringVector &parameterLabels,
                        const tuningParametersMixedPenalty &tuningParameters) override{
                          static_cast<void>(parameterValues); // is unused, but necessary for the interface to be consistent
                          static_cast<void>(parameterLabels); // is unused, but necessary for the interface to be consistent
                          static_cast<void>(tuningParameters); // is unused, but necessary for the interface to be consistent
                               return(0.0);
                             }
};


class penaltyMixedCappedL1: public penaltyMixedPenaltyBase{
public:
  double getValue(const arma::rowvec &parameterValues,
                  const stringVector &parameterLabels,
                  const tuningParametersMixedPenalty &tuningParameters) override{
                               
                               tp.alpha = tuningParameters.alpha(0);
                               tp.lambda = tuningParameters.lambda(0);
                               tp.theta = tuningParameters.theta(0);
                               tp.weights = tuningParameters.weights(0);
                               
                               return(
                                 pen.getValue(
                                   parameterValues,
                                   parameterLabels,
                                   tp
                                 )
                               );
                             }
private: 
  tuningParametersCappedL1 tp;
  penaltyCappedL1 pen;
};

class penaltyMixedLasso: public penaltyMixedPenaltyBase{
public:
  double getValue(const arma::rowvec &parameterValues,
                  const stringVector &parameterLabels,
                  const tuningParametersMixedPenalty &tuningParameters) override{
                               
                               tp.alpha = tuningParameters.alpha(0);
                               tp.lambda = tuningParameters.lambda(0);
                               tp.weights = tuningParameters.weights(0);
                               
                               return(
                                 pen.getValue(
                                   parameterValues,
                                   parameterLabels,
                                   tp
                                 )
                               );
                             }
private: 
  tuningParametersEnet tp;
  penaltyLASSO pen;
};

class penaltyMixedLsp: public penaltyMixedPenaltyBase{
public:
  double getValue(const arma::rowvec &parameterValues,
                  const stringVector &parameterLabels,
                  const tuningParametersMixedPenalty &tuningParameters) override{
                               
                               tp.lambda = tuningParameters.lambda(0);
                               tp.theta = tuningParameters.theta(0);
                               tp.weights = tuningParameters.weights(0);
                               
                               return(
                                 pen.getValue(
                                   parameterValues,
                                   parameterLabels,
                                   tp
                                 )
                               );
                             }
private: 
  tuningParametersLSP tp;
  penaltyLSP pen;
};

class penaltyMixedMcp: public penaltyMixedPenaltyBase{
public:
  double getValue(const arma::rowvec &parameterValues,
                  const stringVector &parameterLabels,
                  const tuningParametersMixedPenalty &tuningParameters) override{
                               
                               tp.lambda = tuningParameters.lambda(0);
                               tp.theta = tuningParameters.theta(0);
                               tp.weights = tuningParameters.weights(0);
                               
                               return(
                                 pen.getValue(
                                   parameterValues,
                                   parameterLabels,
                                   tp
                                 )
                               );
                             }
private: 
  tuningParametersMcp tp;
  penaltyMcp pen;
};


class penaltyMixedScad: public penaltyMixedPenaltyBase{
public:
  double getValue(const arma::rowvec &parameterValues,
                  const stringVector &parameterLabels,
                  const tuningParametersMixedPenalty &tuningParameters) override{
                               
                               tp.lambda = tuningParameters.lambda(0);
                               tp.theta = tuningParameters.theta(0);
                               tp.weights = tuningParameters.weights(0);
                               
                               return(
                                 pen.getValue(
                                   parameterValues,
                                   parameterLabels,
                                   tp
                                 )
                               );
                             }
private: 
  tuningParametersScad tp;
  penaltyScad pen;
};

class penaltyMixedPenalty: public penalty<tuningParametersMixedPenalty> {
public:
  std::vector<std::unique_ptr<penaltyMixedPenaltyBase>> penalties;
  
  double getValue(const arma::rowvec &parameterValues,
                  const stringVector &parameterLabels,
                  const tuningParametersMixedPenalty &tuningParameters) override{
        
        double penaltyValue = 0.0;
        
        arma::rowvec parameterValue{0};
        arma::rowvec parameters_kp1 = parameterValues;
        
        int it = 0;
        for(auto& pen: penalties){
          tpSinglePenalty.alpha = tuningParameters.alpha(it);
          tpSinglePenalty.lambda = tuningParameters.lambda(it);
          tpSinglePenalty.theta = tuningParameters.theta(it);
          tpSinglePenalty.weights = tuningParameters.weights(it);
          
          parameterValue(0) = parameterValues(it);
          
          penaltyValue += arma::as_scalar(pen->getValue(
            parameterValue,
            parameterLabels,
            tpSinglePenalty)
          );
          it++;
        }
        
        return(penaltyValue);
        
      }
  
private:
  tuningParametersMixedPenalty tpSinglePenalty;
};

void inline initializeMixedProximalOperators(proximalOperatorMixedPenalty& proxOperators, 
                                     const std::vector<penaltyType>& penaltyTypes){
  
  for(penaltyType pt: penaltyTypes){
    switch (pt)
    {
    case penaltyType::none:
    {
      std::unique_ptr<proximalOperatorMixedNone> currentPen = std::make_unique<proximalOperatorMixedNone>();
      proxOperators.proxOps.emplace_back(std::move(currentPen));
      break;
    }
    case penaltyType::cappedL1:
    {
      std::unique_ptr<proximalOperatorMixedCappedL1> currentPen = std::make_unique<proximalOperatorMixedCappedL1>();
      proxOperators.proxOps.emplace_back(std::move(currentPen));
      break;
    }
    case penaltyType::lasso:
    {
      std::unique_ptr<proximalOperatorMixedLasso> currentPen = std::make_unique<proximalOperatorMixedLasso>();
      proxOperators.proxOps.emplace_back(std::move(currentPen));
      break;
    }
    case penaltyType::lsp:
    {
      std::unique_ptr<proximalOperatorMixedLsp> currentPen = std::make_unique<proximalOperatorMixedLsp>();
      proxOperators.proxOps.emplace_back(std::move(currentPen));
      break;
    }
    case penaltyType::mcp:
    {
      std::unique_ptr<proximalOperatorMixedMcp> currentPen = std::make_unique<proximalOperatorMixedMcp>();
      proxOperators.proxOps.emplace_back(std::move(currentPen));
      break;
    }
    case penaltyType::scad:
    {
      std::unique_ptr<proximalOperatorMixedScad> currentPen = std::make_unique<proximalOperatorMixedScad>();
      proxOperators.proxOps.emplace_back(std::move(currentPen));
      break;
    }
    default:
      error("Unknown penalty");
    }
  }
}

void inline initializeMixedPenalties(penaltyMixedPenalty& pen, 
                                     const std::vector<penaltyType>& penaltyTypes){
  
  for(penaltyType pt: penaltyTypes){
    switch (pt)
    {
    case penaltyType::none:
    {
      std::unique_ptr<penaltyMixedNone> currentPen = std::make_unique<penaltyMixedNone>();
      pen.penalties.emplace_back(std::move(currentPen));
      break;
    }
    case penaltyType::cappedL1:
    {
      std::unique_ptr<penaltyMixedCappedL1> currentPen = std::make_unique<penaltyMixedCappedL1>();
      pen.penalties.emplace_back(std::move(currentPen));
      break;
    }
    case penaltyType::lasso:
    {
      std::unique_ptr<penaltyMixedLasso> currentPen = std::make_unique<penaltyMixedLasso>();
      pen.penalties.emplace_back(std::move(currentPen));
      break;
    }
    case penaltyType::lsp:
    {
      std::unique_ptr<penaltyMixedLsp> currentPen = std::make_unique<penaltyMixedLsp>();
      pen.penalties.emplace_back(std::move(currentPen));
      break;
    }
    case penaltyType::mcp:
    {
      std::unique_ptr<penaltyMixedMcp> currentPen = std::make_unique<penaltyMixedMcp>();
      pen.penalties.emplace_back(std::move(currentPen));
      break;
    }
    case penaltyType::scad:
    {
      std::unique_ptr<penaltyMixedScad> currentPen = std::make_unique<penaltyMixedScad>();
      pen.penalties.emplace_back(std::move(currentPen));
      break;
    }
    default:
      error("Unknown penalty");
    }
  }
}

}
#endif
