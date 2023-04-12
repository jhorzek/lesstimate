#ifndef SCAD_GLMNET_H
#define SCAD_GLMNET_H

#include "penalty.h"

namespace lessSEM
{

    class tuningParametersScadGlmnet
    {
    public:
        arma::rowvec weights;
        double lambda;
        double theta;
    };

    // the glmnet penalty allows for vectors of alpha and lambda
    class penaltySCADGlmnet : public penalty<tuningParametersScadGlmnet>
    {
    public:
        double getValue(const arma::rowvec &parameterValues,
                        const Rcpp::StringVector &parameterLabels,
                        const tuningParametersScadGlmnet &tuningParameters)
            override
        {

            double penalty = 0.0;

            for (unsigned int p = 0; p < parameterValues.n_elem; p++)
            {

                if (tuningParameters.weights.at(p) == 0)
                    continue;

                double lambda = tuningParameters.weights.at(p) * tuningParameters.lambda;
                double theta = tuningParameters.theta;

                double absPar = std::abs(parameterValues.at(p));

                if (absPar <= lambda)
                {
                    // reduces to lasso penalty
                    penalty += (lambda * absPar);
                }
                else if ((lambda < absPar) && (absPar <= lambda * theta))
                {
                    // reduces to a smooth penalty
                    penalty += ((-std::pow(parameterValues.at(p), 2) +
                                 2.0 * theta * lambda * absPar - std::pow(lambda, 2)) /
                                (2.0 * (theta - 1.0)));
                }
                else if (absPar > (lambda * theta))
                {
                    // reduces to a constant penalty
                    penalty += (((theta + 1.0) * std::pow(lambda, 2)) / 2.0);
                }
                else
                {
                    // the following should never be called:
                    Rcpp::stop("Error while evaluating scad");
                }
            }

            return penalty;
        }

        // subproblemValue
        //
        // glmnet uses a combination of inner and outer iterations. Within the inner iteration, a
        // subproblem is solved for a single parameter. The scad penalty is non-convex which
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

            if (probe <= lambda)
                return (base + lambda * probe);

            if ((lambda < probe) && (probe <= lambda * theta))
                return (base + (-probe * probe + 2.0 * theta * lambda * probe - lambda * lambda) / (2.0 * (theta - 1.0)));

            if (probe >= lambda * theta)
                return (base + (theta + 1.0) * lambda * lambda / 2.0);

            Rcpp::stop("This should not have happened... Scad ran into issues");
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
            const tuningParametersScadGlmnet &tuningParameters)
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

            // The scad penalty is non-convex and may have multiple minima.
            // However, there are parts of the function that are convex. We
            // can therefore check these parts for their respective minima and
            // then check which is the overall minimum.
            double z[5];
            double fitValue[5];

            // Case 1: lasso
            double probe1 = parameterValue_j + d_j - (g_j + hessianXdirection_j + lambda) / H_jj;
            double probe2 = parameterValue_j + d_j - (g_j + hessianXdirection_j - lambda) / H_jj;

            if ((probe1 > 0))
            {
                // parameterValue_j + d_j + z > 0
                // additionally: parameterValue_j + d_j + z < lambda
                // so z must be z < lambda - (parameterValue_j + d_j)
                z[0] = std::min(lambda - (parameterValue_j + d_j),
                                -(g_j + hessianXdirection_j + lambda) / H_jj);
            }
            else if ((probe2 < 0))
            {
                // parameterValue_j + d_j + z < 0
                // additionally: parameterValue_j + d_j + z < -lambda
                // so z must be z < -lambda - (parameterValue_j + d_j)
                z[0] = std::max(-lambda - (parameterValue_j + d_j),
                                -(g_j + hessianXdirection_j - lambda) / H_jj);
            }
            else
            {
                // parameterValue_j + d_j + z = 0
                // it directly follows that -lambda <= parameterValue_j + d_j + z <= lambda
                z[0] = -parameterValue_j - d_j;
            }

            // Case 2: smooth penalty
            // assume that parameterValue_j + d_j + z > 0
            // additionally: parameterValue_j + d_j + z >  lambda       -> z >  lambda - (parameterValue_j + d_j)
            // and:          parameterValue_j + d_j + z <= lambda*theta -> z <= lambda*theta - (parameterValue_j + d_j)
            z[1] = std::max(
                lambda - (parameterValue_j + d_j),
                std::min(
                    lambda * theta - (parameterValue_j + d_j),
                    (parameterValue_j + d_j - theta * lambda - (g_j + hessianXdirection_j) * (theta - 1)) / (H_jj * (theta - 1) - 1)));

            // assume that parameterValue_j + d_j + z < 0
            // additionally: parameterValue_j + d_j + z <  -lambda       -> z <  -lambda - (parameterValue_j + d_j)
            // and:          parameterValue_j + d_j + z >= -lambda*theta -> z >= -lambda*theta - (parameterValue_j + d_j)
            z[2] = std::max(
                -lambda * theta - (parameterValue_j + d_j),
                std::min(
                    -lambda - (parameterValue_j + d_j),
                    (parameterValue_j + d_j + theta * lambda - (g_j + hessianXdirection_j) * (theta - 1)) / (H_jj * (theta - 1) - 1)));

            // Case 3: constant penalty
            //     parameterValue_j + d_j + z >  lambda*theta -> z >  lambda*theta - (parameterValue_j + d_j)
            // or: parameterValue_j + d_j + z < -lambda*theta -> z < -lambda*theta - (parameterValue_j + d_j)
            // if parameterValue_j + d_j + z is positive:
            z[3] = std::max(lambda * theta - (parameterValue_j + d_j),
                            -(g_j + hessianXdirection_j) / H_jj);
            // if parameterValue_j + d_j + z is negative:
            z[4] = std::min(-lambda * theta - (parameterValue_j + d_j),
                            -(g_j + hessianXdirection_j) / H_jj);

            // compute fit value
            int whichmin = 0;
            for (unsigned int i = 0; i < 5; i++)
            {

                fitValue[i] = this->subproblemValue(
                    parameterValue_j,
                    z[i],
                    g_j,
                    d_j,
                    hessianXdirection_j,
                    H_jj,
                    lambda,
                    theta);

                if (i > 0)
                {
                    if (fitValue[i] < fitValue[whichmin])
                        whichmin = i;
                }
            }

            return (z[whichmin]);
        }

        arma::rowvec getSubgradients(const arma::rowvec &parameterValues,
                                     const arma::rowvec &gradients,
                                     const tuningParametersScadGlmnet &tuningParameters)
        {
            Rcpp::stop("Subgradients not yet implemented for scad penalty. Use different convergence criterion.");
        }
    };

}

#endif