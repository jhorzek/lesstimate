#ifndef MIXEDPENALTY_GLMNET_H
#define MIXEDPENALTY_GLMNET_H
#include <memory>
#include "common_headers.h"

#include "penalty.h"
#include "penalty_type.h"
#include "enet.h"
#include "glmnet_cappedL1.h"
#include "glmnet_lasso.h"
#include "glmnet_lsp.h"
#include "glmnet_mcp.h"
#include "glmnet_scad.h"

namespace lessSEM
{

  // We want to allow users to combine different penalties. To this end,
  // we create a new class of tuning parameters that has all possible options:
  /**
   * @brief tuning parameters for the mixed penalty optimized with glmnet
   *
   */
  class tuningParametersMixedGlmnet
  {
  public:
    std::vector<penaltyType> penaltyType_; ///> penaltyType-vector specifying the penalty to be used for each parameter
    arma::rowvec lambda;                   ///> provide parameter-specific lambda values
    arma::rowvec theta;                    ///> theta value of the mixed penalty > 0
    arma::rowvec alpha;                    ///> alpha value of the mixed penalty > 0
    arma::rowvec weights;                  ///> provide parameter-specific weights (e.g., for adaptive lasso)
  };


  /**
  * @brief base class for mixed penalty for glmnet optimizer
  *
  */
  class penaltyMixedGlmnetBase{
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
                            const tuningParametersMixedGlmnet &tuningParameters) = 0;
    
    /**
     * @brief computes the step direction for a single parameter j in the inner
     * iterations of the lasso penalty.
     *
     * @param whichPar index of parameter j
     * @param parameters_kMinus1 parameter values at previous iteration
     * @param gradient gradients of fit function
     * @param stepDirection step direction
     * @param Hessian Hessian matrix
     * @param tuningParameters tuning parameters
     * @return double step direction for parameter j
     */
    virtual double getZ(
        unsigned int whichPar,
        const arma::rowvec &parameters_kMinus1,
        const arma::rowvec &gradient,
        const arma::rowvec &stepDirection,
        const arma::mat &Hessian,
        const tuningParametersMixedGlmnet &tuningParameters) = 0;
    
    
    /**
     * @brief Get the subgradients of the penalty function
     *
     * @param parameterValues current parameter values
     * @param parameterLabels names of the parameters
     * @param tuningParameters values of the tuning parmameters
     * @return arma::rowvec
     */
    arma::rowvec getSubgradients(const arma::rowvec &parameterValues,
                                 const arma::rowvec &gradients,
                                 const tuningParametersMixedGlmnet &tuningParameters)
    {

      static_cast<void>(parameterValues); // is unused
      static_cast<void>(gradients); // is unused
      static_cast<void>(tuningParameters); // is unused
      error("Subgradients are not yet implemented for mixedPenalty");
    }
    
    /**
     * @brief Check the dimensions of the tuning parameters
     *
     * @return throws error in case of false dimensions
     */
    void checkDimensions(const tuningParametersMixedGlmnet &tuningParameters){
      if(!checked){
        // check the dimensions on the first call
        if(tuningParameters.alpha.n_elem > 0)
          error("Incorrect length of tuning parameters");
        if(tuningParameters.lambda.n_elem > 0)
          error("Incorrect length of tuning parameters");
        if(tuningParameters.weights.n_elem > 0)
          error("Incorrect length of tuning parameters");
        
        checked = true;
      }
    }
    
    bool checked = false;
  };
  
  // implementations
  class penaltyMixedGlmnetNone: public penaltyMixedGlmnetBase{
    
    double getValue(const arma::rowvec &parameterValues,
                    const stringVector &parameterLabels,
                    const tuningParametersMixedGlmnet &tuningParameters) override {
                      static_cast<void>(parameterValues); // is unused
                      static_cast<void>(parameterLabels); // is unused
                      static_cast<void>(tuningParameters); // is unused
                      return(0.0);
                    }
    
    double getZ(
        unsigned int whichPar,
        const arma::rowvec &parameters_kMinus1,
        const arma::rowvec &gradient,
        const arma::rowvec &stepDirection,
        const arma::mat &Hessian,
        const tuningParametersMixedGlmnet &tuningParameters) override{

          static_cast<void>(parameters_kMinus1); // is unused, but necessary for the interface to be consistent
          static_cast<void>(tuningParameters); // is unused, but necessary for the interface to be consistent
          
          arma::colvec hessianXdirection = Hessian * arma::trans(stepDirection);
          double hessianXdirection_j = arma::as_scalar(hessianXdirection.row(whichPar));
          double H_jj = arma::as_scalar(Hessian.row(whichPar).col(whichPar));
          double g_j = arma::as_scalar(gradient.col(whichPar));
          
          return (-(g_j + hessianXdirection_j) / H_jj);
          
        }
  };
  
  
  class penaltyMixedGlmnetCappedL1: public penaltyMixedGlmnetBase{
    penaltyCappedL1Glmnet pen;
    tuningParametersCappedL1Glmnet tp;
    
    double getValue(const arma::rowvec &parameterValues,
                    const stringVector &parameterLabels,
                    const tuningParametersMixedGlmnet &tuningParameters) override {
                      tp.lambda = tuningParameters.lambda(0);
                      tp.theta = tuningParameters.theta(0);
                      tp.weights = tuningParameters.weights(0);
                      return(pen.getValue(parameterValues, parameterLabels, tp));
                    }
    
    double getZ(
        unsigned int whichPar,
        const arma::rowvec &parameters_kMinus1,
        const arma::rowvec &gradient,
        const arma::rowvec &stepDirection,
        const arma::mat &Hessian,
        const tuningParametersMixedGlmnet &tuningParameters) override{
          
          tp.lambda = tuningParameters.lambda(whichPar);
          tp.theta = tuningParameters.theta(whichPar);
          tp.weights = tuningParameters.weights;
          
          return(pen.getZ(whichPar,
                          parameters_kMinus1,
                          gradient,
                          stepDirection,
                          Hessian,
                          tp));
        }
  };
  
  class penaltyMixedGlmnetLasso: public penaltyMixedGlmnetBase{
    penaltyLASSOGlmnet pen;
    tuningParametersEnetGlmnet tp;
    
    double getValue(const arma::rowvec &parameterValues,
                    const stringVector &parameterLabels,
                    const tuningParametersMixedGlmnet &tuningParameters) override {
                      tp.alpha = tuningParameters.alpha(0);
                      tp.lambda = tuningParameters.lambda(0);
                      tp.weights = tuningParameters.weights(0);
                      return(pen.getValue(parameterValues, parameterLabels, tp));
                    }
    
    double getZ(
        unsigned int whichPar,
        const arma::rowvec &parameters_kMinus1,
        const arma::rowvec &gradient,
        const arma::rowvec &stepDirection,
        const arma::mat &Hessian,
        const tuningParametersMixedGlmnet &tuningParameters) override{
          
          tp.alpha = tuningParameters.alpha;
          tp.lambda = tuningParameters.lambda;
          tp.weights = tuningParameters.weights;
          
          return(pen.getZ(whichPar,
                          parameters_kMinus1,
                          gradient,
                          stepDirection,
                          Hessian,
                          tp));
        }
  };
  
  class penaltyMixedGlmnetLsp: public penaltyMixedGlmnetBase{
    penaltyLSPGlmnet pen;
    tuningParametersLspGlmnet tp;
    
    double getValue(const arma::rowvec &parameterValues,
                    const stringVector &parameterLabels,
                    const tuningParametersMixedGlmnet &tuningParameters) override {
                      tp.lambda = tuningParameters.lambda(0);
                      tp.theta = tuningParameters.theta(0);
                      tp.weights = tuningParameters.weights(0);
                      return(pen.getValue(parameterValues, parameterLabels, tp));
                    }
    
    double getZ(
        unsigned int whichPar,
        const arma::rowvec &parameters_kMinus1,
        const arma::rowvec &gradient,
        const arma::rowvec &stepDirection,
        const arma::mat &Hessian,
        const tuningParametersMixedGlmnet &tuningParameters) override{
          
          tp.lambda = tuningParameters.lambda(whichPar);
          tp.theta = tuningParameters.theta(whichPar);
          tp.weights = tuningParameters.weights;
          
          return(pen.getZ(whichPar,
                          parameters_kMinus1,
                          gradient,
                          stepDirection,
                          Hessian,
                          tp));
        }
  };
  
  class penaltyMixedGlmnetMcp: public penaltyMixedGlmnetBase{
    penaltyMcpGlmnet pen;
    tuningParametersMcpGlmnet tp;
    
    double getValue(const arma::rowvec &parameterValues,
                    const stringVector &parameterLabels,
                    const tuningParametersMixedGlmnet &tuningParameters) override {
                      tp.lambda = tuningParameters.lambda(0);
                      tp.theta = tuningParameters.theta(0);
                      tp.weights = tuningParameters.weights(0);
                      return(pen.getValue(parameterValues, parameterLabels, tp));
                    }
    
    double getZ(
        unsigned int whichPar,
        const arma::rowvec &parameters_kMinus1,
        const arma::rowvec &gradient,
        const arma::rowvec &stepDirection,
        const arma::mat &Hessian,
        const tuningParametersMixedGlmnet &tuningParameters) override{
          
          tp.lambda = tuningParameters.lambda(whichPar);
          tp.theta = tuningParameters.theta(whichPar);
          tp.weights = tuningParameters.weights;
          
          return(pen.getZ(whichPar,
                          parameters_kMinus1,
                          gradient,
                          stepDirection,
                          Hessian,
                          tp));
        }
  };
  
  class penaltyMixedGlmnetScad: public penaltyMixedGlmnetBase{
    penaltySCADGlmnet pen;
    tuningParametersScadGlmnet tp;
    
    double getValue(const arma::rowvec &parameterValues,
                    const stringVector &parameterLabels,
                    const tuningParametersMixedGlmnet &tuningParameters) override {
                      tp.lambda = tuningParameters.lambda(0);
                      tp.theta = tuningParameters.theta(0);
                      tp.weights = tuningParameters.weights(0);
                      return(pen.getValue(parameterValues, parameterLabels, tp));
                    }
    
    double getZ(
        unsigned int whichPar,
        const arma::rowvec &parameters_kMinus1,
        const arma::rowvec &gradient,
        const arma::rowvec &stepDirection,
        const arma::mat &Hessian,
        const tuningParametersMixedGlmnet &tuningParameters) override{
          
          tp.lambda = tuningParameters.lambda(whichPar);
          tp.theta = tuningParameters.theta(whichPar);
          tp.weights = tuningParameters.weights;
          
          return(pen.getZ(whichPar,
                          parameters_kMinus1,
                          gradient,
                          stepDirection,
                          Hessian,
                          tp));
        }
  };
  
  class penaltyMixedGlmnet: public penalty<tuningParametersMixedGlmnet>{
    
  public:
    std::vector<std::unique_ptr<penaltyMixedGlmnetBase>> penalties;
    
    /**
     * @brief Get the value of the penalty function
     *
     * @param parameterValues current parameter values
     * @param parameterLabels names of the parameters
     * @param tuningParameters values of the tuning parmameters
     * @return double
     */
    double getValue(const arma::rowvec &parameterValues,
                    const stringVector &parameterLabels,
                    const tuningParametersMixedGlmnet &tuningParameters)
    override
    {
      double penVal{0.0};
      int it = 0;
      for(auto& pen: penalties){
        tpSinglePenalty.alpha = tuningParameters.alpha(it);
        tpSinglePenalty.lambda = tuningParameters.lambda(it);
        tpSinglePenalty.theta = tuningParameters.theta(it);
        tpSinglePenalty.weights = tuningParameters.weights(it);
        
        arma::rowvec parameterValue(1);
        parameterValue(0) = parameterValues(it);
        stringVector parameterLabel(1);
        parameterLabel.at(0) = parameterLabels.at(it);
        
        penVal += pen->getValue(parameterValue,
                                parameterLabel,
                                tpSinglePenalty);
        it++;
      }
      
      return(penVal);
    }
    
    /**
     * @brief computes the step direction for a single parameter j in the inner
     * iterations of the lasso penalty.
     *
     * @param whichPar index of parameter j
     * @param parameters_kMinus1 parameter values at previous iteration
     * @param gradient gradients of fit function
     * @param stepDirection step direction
     * @param Hessian Hessian matrix
     * @param tuningParameters tuning parameters
     * @return double step direction for parameter j
     */
    double getZ(
        unsigned int whichPar,
        const arma::rowvec &parameters_kMinus1,
        const arma::rowvec &gradient,
        const arma::rowvec &stepDirection,
        const arma::mat &Hessian,
        const tuningParametersMixedGlmnet &tuningParameters)
    {
      
      tpSinglePenalty.alpha = tuningParameters.alpha;
      tpSinglePenalty.lambda = tuningParameters.lambda;
      tpSinglePenalty.theta = tuningParameters.theta;
      tpSinglePenalty.weights = tuningParameters.weights;
      
      double z = penalties.at(whichPar)->getZ(whichPar,
                                      parameters_kMinus1,
                                      gradient,
                                      stepDirection,
                                      Hessian,
                                      tpSinglePenalty);
        
      return(z);
      
    }
    
    /**
     * @brief Get the subgradients of the penalty function
     *
     * @param parameterValues current parameter values
     * @param parameterLabels names of the parameters
     * @param tuningParameters values of the tuning parmameters
     * @return arma::rowvec
     */
    arma::rowvec getSubgradients(const arma::rowvec &parameterValues,
                                 const arma::rowvec &gradients,
                                 const tuningParametersMixedGlmnet &tuningParameters)
    {
      static_cast<void>(parameterValues); // is unused
      static_cast<void>(gradients); // is unused
      static_cast<void>(tuningParameters); // is unused
      error("Subgradients are not yet implemented for mixedPenalty");
    }
    
  private:
    // we often need the tuning parameters for a single parameter. These are
    // stored here:
    tuningParametersMixedGlmnet tpSinglePenalty;
    
  };
  
  void inline initializeMixedPenaltiesGlmnet(penaltyMixedGlmnet& pen, 
                                      const std::vector<penaltyType>& penaltyTypes){
    
    for(penaltyType pt: penaltyTypes){
      switch (pt)
      {
      case penaltyType::none:
        {
          std::unique_ptr<penaltyMixedGlmnetNone> currentPen = std::make_unique<penaltyMixedGlmnetNone>();
          pen.penalties.emplace_back(std::move(currentPen));
          break;
        }
      case penaltyType::cappedL1:
        {
          std::unique_ptr<penaltyMixedGlmnetCappedL1> currentPen = std::make_unique<penaltyMixedGlmnetCappedL1>();
          pen.penalties.emplace_back(std::move(currentPen));
          break;
        }
      case penaltyType::lasso:
      {
        std::unique_ptr<penaltyMixedGlmnetLasso> currentPen = std::make_unique<penaltyMixedGlmnetLasso>();
        pen.penalties.emplace_back(std::move(currentPen));
        break;
      }
      case penaltyType::lsp:
      {
        std::unique_ptr<penaltyMixedGlmnetLsp> currentPen = std::make_unique<penaltyMixedGlmnetLsp>();
        pen.penalties.emplace_back(std::move(currentPen));
        break;
      }
      case penaltyType::mcp:
      {
        std::unique_ptr<penaltyMixedGlmnetMcp> currentPen = std::make_unique<penaltyMixedGlmnetMcp>();
        pen.penalties.emplace_back(std::move(currentPen));
        break;
      }
      case penaltyType::scad:
      {
        std::unique_ptr<penaltyMixedGlmnetScad> currentPen = std::make_unique<penaltyMixedGlmnetScad>();
        pen.penalties.emplace_back(std::move(currentPen));
        break;
      }
      default:
        error("Unknown penalty");
      }
    }
  }
} // end namespace
#endif