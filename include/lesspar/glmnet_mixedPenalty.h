#ifndef MIXEDPENALTY_GLMNET_H
#define MIXEDPENALTY_GLMNET_H
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

  // The following is going to be a lot of copy paste from the specific penalty
  // functions. Until I've found a better way to

  /**
   * @brief mixed penalty for glmnet optimizer
   *
   */
  class penaltyMixedGlmnet : public penalty<tuningParametersMixedGlmnet>
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
    double getValue(const arma::rowvec &parameterValues,
                    const stringVector &parameterLabels,
                    const tuningParametersMixedGlmnet &tuningParameters)
        override
    {

      double penalty = 0.0;
      arma::rowvec parameterValue(1);
      stringVector parameterLabel(1);

      // The following is really ugly, but it's easy to implement, so here we go...
      tuningParametersCappedL1Glmnet tpCappedL1;
      penaltyCappedL1Glmnet penCappedL1;

      tuningParametersEnetGlmnet tpEnet;
      penaltyLASSOGlmnet penLasso;

      tuningParametersLspGlmnet tpLsp;
      penaltyLSPGlmnet penLsp;

      tuningParametersMcpGlmnet tpMcp;
      penaltyMcpGlmnet penMcp;

      tuningParametersScadGlmnet tpScad;
      penaltySCADGlmnet penScad;

      for (unsigned int p = 0; p < parameterValues.n_elem; p++)
      {

        penaltyType pt = tuningParameters.penaltyType_.at(p);

        switch (pt)
        {

        case none:
          break;

        case cappedL1:

          tpCappedL1.lambda = tuningParameters.lambda.at(p);
          tpCappedL1.theta = tuningParameters.theta.at(p);
          tpCappedL1.weights = tuningParameters.weights.at(p);

          parameterValue.col(0) = parameterValues.col(p);
          parameterLabel.at(0) = parameterLabels.at(p);

          penalty += penCappedL1.getValue(parameterValue, parameterLabel, tpCappedL1);

          break;

        case lasso:

          tpEnet.lambda = tuningParameters.lambda.at(p);
          tpEnet.alpha = tuningParameters.alpha.at(p);
          tpEnet.weights = tuningParameters.weights.at(p);

          parameterValue.col(0) = parameterValues.col(p);
          parameterLabel.at(0) = parameterLabels.at(p);

          penalty += penLasso.getValue(parameterValue, parameterLabel, tpEnet);

          break;

        case lsp:

          tpLsp.lambda = tuningParameters.lambda.at(p);
          tpLsp.theta = tuningParameters.theta.at(p);
          tpLsp.weights = tuningParameters.weights.at(p);

          parameterValue.col(0) = parameterValues.col(p);
          parameterLabel.at(0) = parameterLabels.at(p);

          penalty += penLsp.getValue(parameterValue, parameterLabel, tpLsp);

          break;

        case mcp:

          tpMcp.lambda = tuningParameters.lambda.at(p);
          tpMcp.theta = tuningParameters.theta.at(p);
          tpMcp.weights = tuningParameters.weights.at(p);

          parameterValue.col(0) = parameterValues.col(p);
          parameterLabel.at(0) = parameterLabels.at(p);

          penalty += penMcp.getValue(parameterValue, parameterLabel, tpMcp);

          break;

        case scad:

          tpScad.lambda = tuningParameters.lambda.at(p);
          tpScad.theta = tuningParameters.theta.at(p);
          tpScad.weights = tuningParameters.weights.at(p);

          parameterValue.col(0) = parameterValues.col(p);
          parameterLabel.at(0) = parameterLabels.at(p);

          penalty += penScad.getValue(parameterValue, parameterLabel, tpScad);

          break;

        default:
          error("Unknown penalty type");
        }
      }

      return penalty;
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
      // The following is really ugly, but it's easy to implement, so here we go...

      tuningParametersCappedL1Glmnet tpCappedL1;
      penaltyCappedL1Glmnet penCappedL1;

      tuningParametersEnetGlmnet tpEnet;
      penaltyLASSOGlmnet penLasso;

      tuningParametersLspGlmnet tpLsp;
      penaltyLSPGlmnet penLsp;

      tuningParametersMcpGlmnet tpMcp;
      penaltyMcpGlmnet penMcp;

      tuningParametersScadGlmnet tpScad;
      penaltySCADGlmnet penScad;

      penaltyType pt = tuningParameters.penaltyType_.at(whichPar);

      switch (pt)
      {

      case none:
      {
        arma::colvec hessianXdirection = Hessian * arma::trans(stepDirection);
        double hessianXdirection_j = arma::as_scalar(hessianXdirection.row(whichPar));
        double H_jj = arma::as_scalar(Hessian.row(whichPar).col(whichPar));
        double g_j = arma::as_scalar(gradient.col(whichPar));

        return (-(g_j + hessianXdirection_j) / H_jj);
      }

      case cappedL1:

        tpCappedL1.lambda = tuningParameters.lambda.at(whichPar);
        tpCappedL1.theta = tuningParameters.theta.at(whichPar);
        tpCappedL1.weights = tuningParameters.weights;

        return (penCappedL1.getZ(
            whichPar,
            parameters_kMinus1,
            gradient,
            stepDirection,
            Hessian,
            tpCappedL1));

      case lasso:

        tpEnet.lambda = tuningParameters.lambda;
        tpEnet.alpha = tuningParameters.alpha;
        tpEnet.weights = tuningParameters.weights;

        return (penLasso.getZ(
            whichPar,
            parameters_kMinus1,
            gradient,
            stepDirection,
            Hessian,
            tpEnet));

      case lsp:

        tpLsp.lambda = tuningParameters.lambda.at(whichPar);
        tpLsp.theta = tuningParameters.theta.at(whichPar);
        tpLsp.weights = tuningParameters.weights;

        return (penLsp.getZ(
            whichPar,
            parameters_kMinus1,
            gradient,
            stepDirection,
            Hessian,
            tpLsp));

      case mcp:

        tpMcp.lambda = tuningParameters.lambda.at(whichPar);
        tpMcp.theta = tuningParameters.theta.at(whichPar);
        tpMcp.weights = tuningParameters.weights;

        return (penMcp.getZ(
            whichPar,
            parameters_kMinus1,
            gradient,
            stepDirection,
            Hessian,
            tpMcp));

      case scad:

        tpScad.lambda = tuningParameters.lambda.at(whichPar);
        tpScad.theta = tuningParameters.theta.at(whichPar);
        tpScad.weights = tuningParameters.weights;

        return (penScad.getZ(
            whichPar,
            parameters_kMinus1,
            gradient,
            stepDirection,
            Hessian,
            tpScad));

      default:
        error("Unknown penalty type");
      }
      error("Unknown penalty type");

    } // end getZ

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
      error("Subgradients are not yet implemented for mixedPenalty");
    }
  };

}
#endif