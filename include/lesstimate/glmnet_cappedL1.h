#ifndef GLMNET_CAPPEDL1
#define GLMNET_CAPPEDL1

#include "penalty.h"
#include "common_headers.h"

namespace lessSEM
{
    /**
     * @brief tuning parameters for the capped L1 penalty optimized with glmnet
     *
     */
    class tuningParametersCappedL1Glmnet
    {
    public:
        arma::rowvec weights; ///> provide parameter-specific weights (e.g., for adaptive lasso)
        double lambda;        ///> lambda value >= 0
        double theta;         ///> theta value of the cappedL1 penalty > 0
    };

    /**
     * @brief cappedL1 penalty for glmnet optimizer.
     * 
     * The penalty function is given by:
     * $$p( x_j) = \lambda \min(| x_j|, \theta)$$
     * where $\theta > 0$. The cappedL1 penalty is identical to the lasso for
     * parameters which are below $\theta$ and identical to a constant for parameters
     * above $\theta$. As adding a constant to the fitting function will not change its
     * minimum, larger parameters can stay unregularized while smaller ones are set to zero.
     *
     * CappedL1 regularization:
     *
     * * Zhang, T. (2010). Analysis of Multi-stage Convex Relaxation for Sparse Regularization.
     * Journal of Machine Learning Research, 11, 1081–1107.
     */
    class penaltyCappedL1Glmnet : public penalty<tuningParametersCappedL1Glmnet>
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
                        const tuningParametersCappedL1Glmnet &tuningParameters)
            override
        {

            static_cast<void>(parameterLabels); // is unused, but necessary for the interface to be consistent

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

        /**
         * @brief glmnet uses a combination of inner and outer iterations. Within the inner iteration, a
         * subproblem is solved for a single parameter. The cappedL1 penalty is non-convex which
         * means that there may be local minima in the subproblem. However, because the function is
         * convex within regions, we can find the minimum within each region and then compare the results
         * to find the global minimum. To this end, we need the function value of the subproblem. This
         * is computed here.
         * @param parameterValue_j parameter value from the outer iteration for parameter j
         * @param z update for parameter j in current inner iteration
         * @param g_j gradient value from the outer iteration for parameter j
         * @param d_j direction value from the inner iteration for parameter j
         * @param hessianXdirection_j product of hessian and direction parameter value from the outer iteration for parameter j
         * @param H_jj row j, col j of Hessian matrix
         * @param lambda tuning parameter lambda
         * @param theta tuning parameter theta
         * @return fit value (double)
         */
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
            z[1] = (-g_j - hessianXdirection_j) / H_jj;

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
                                     const tuningParametersCappedL1Glmnet &tuningParameters)
        {
            static_cast<void>(parameterValues); // is unused
            static_cast<void>(gradients); // is unused
            static_cast<void>(tuningParameters); // is unused
            error("Subgradients not yet implemented for cappedL1 penalty. Use different convergence criterion.");
        }
    };
}

#endif