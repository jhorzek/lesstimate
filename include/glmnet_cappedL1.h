#ifndef GLMNET_CAPPEDL1
#define GLMNET_CAPPEDL1

#include "penalty.h"

namespace lessSEM
{

    class tuningParametersCappedL1Glmnet
    {
    public:
        arma::rowvec weights;
        double lambda;
        double theta;
    };

    class penaltyCappedL1Glmnet : public penalty<tuningParametersCappedL1Glmnet>
    {
    public:
        double getValue(const arma::rowvec &parameterValues,
                        const Rcpp::StringVector &parameterLabels,
                        const tuningParametersCappedL1Glmnet &tuningParameters)
            override
        {

            double penalty = 0.0;
            double lambda_i;
            double theta = tuningParameters.theta;

            for (unsigned int p = 0; p < parameterValues.n_elem; p++)
            {

                if (tuningParameters.weights.at(p) == 0)
                    continue;

                lambda_i = tuningParameters.lambda *
                           tuningParameters.weights.at(p);

                penalty += lambda_i * std::min(std::abs(parameterValues.at(p)),
                                               theta);
            }

            return penalty;
        }

        // subproblemValue
        //
        // glmnet uses a combination of inner and outer iterations. Within the inner iteration, a
        // subproblem is solved for a single parameter. The cappedL1 penalty is non-convex which
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

            return (base + lambda * std::min(theta, std::abs(parameterValue_j + d_j + z)));
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
            const tuningParametersCappedL1Glmnet &tuningParameters)
        {
            double tuning = tuningParameters.weights.at(whichPar) * tuningParameters.lambda;
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

            // CappedL1 is non-convex, but has convex regions. We test 
            // both of these regions to check for the global minimum
            double z[2];
            double fitValue[2];

            // Case 1: standard lasso
            double probe = parameterValue_j + d_j - (g_j + hessianXdirection_j) / H_jj;

            if (probe - tuning / H_jj > 0)
            {
                // parameter is positive and we have to make sure that we stay within the boundaries
                // parameterValue_j + d_j + z < theta -> z < theta - (parameterValue_j + d_j)
              z[0] = std::min(
                    theta - (parameterValue_j + d_j),
                    (-(g_j + hessianXdirection_j + tuning) / H_jj));
            }
            else if (probe + tuning / H_jj < 0)
            {
                // parameter is negative and we have to make sure that we stay within the boundaries
                // parameterValue_j + d_j + z > -theta -> z < -theta - (parameterValue_j + d_j)
              z[0] = std::max(
                    -theta - (parameterValue_j + d_j),
                    -(g_j + hessianXdirection_j - tuning) / H_jj);
            }
            else
            {
                // parameter is zero
                z[0] = -parameterValue_j - d_j;
            }
            
            // assume that |parameterValue_j + d_j + z| > theta
            z[1] = (-g_j - hessianXdirection_j)/H_jj;
            
            // compute fit value
            int whichmin = 0;
            for (unsigned int i = 0; i < 2; i++)
            {

                fitValue[i] = this->subproblemValue(
                    parameterValue_j,
                    z[i],
                    g_j,
                    d_j,
                    hessianXdirection_j,
                    H_jj,
                    tuning,
                    theta);

                if (i > 0)
                {
                    if (fitValue[i] < fitValue[whichmin])
                        whichmin = i;
                }
            }
            return (z[whichmin]);
        }

        arma::rowvec
        getSubgradients(const arma::rowvec &parameterValues,
                        const arma::rowvec &gradients,
                        const tuningParametersCappedL1Glmnet &tuningParameters)
        {
            Rcpp::stop("Subgradients not yet implemented for cappedL1 penalty. Use different convergence criterion.");
        }
    };
}

#endif