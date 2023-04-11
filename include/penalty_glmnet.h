#ifndef PENALTY_GLMNET.h
#define PENALTY_GLMNET .h

#include "penalty.h"

namespace lessSEM
{

    class penaltyGlmnet : public penalty
    {
    public:
        // function to update a single parameter:
        virtual double getZ();
    };

    // the glmnet penalty allows for vectors of alpha and lambda
    class penaltyLASSOGlmnet : public penalty<tuningParametersEnetGlmnet>
    {
    public:
        double getValue(const arma::rowvec &parameterValues,
                        const Rcpp::StringVector &parameterLabels,
                        const tuningParametersEnetGlmnet &tuningParameters)
            override
        {

            double penalty = 0.0;
            double lambda_i;

            for (unsigned int p = 0; p < parameterValues.n_elem; p++)
            {

                lambda_i = tuningParameters.alpha.at(p) *
                           tuningParameters.lambda.at(p) *
                           tuningParameters.weights.at(p);

                penalty += lambda_i * std::abs(parameterValues.at(p));
            }

            return penalty;
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
            unsigned int whichPar;
            const arma::rowvec& gradient,
            const arma::rowvec& stepDirection,
            const arma::mat& Hessian,
            const tuningParametersEnetGlmnet &tuningParameters)
            override
        {

            double tuning = tuningParameters.alpha.at(whichPar) *
                            tuningParameters.lambda.at(whichPar) *
                            tuningParameters.weights.at(whichPar);

            // compute derivative elements:
            arma::rowvec hessianXdirection = Hessian * arma::trans(stepDirection);
            double H_jj = Hessian.row(whichPar).col(whichPar);
            double g_j = gradient.col(whichPar);
            // if the parameter is regularized:
            if (tuning != 0)
            {
                double probe = parameterValue_j + d_j - (g_j + hessianXdirection_j) / H_jj;

                if (probe - tuning / H_jj > 0)
                    return (-(g_j + hessianXdirection_j + tuning) / H_jj);

                if (probe + tuning / H_jj < 0)
                    return (-(g_j + hessianXdirection_j - tuning) / H_jj);

                return (-parameterValue_j - d_j);
            }
            else
            {
                // if not regularized: coordinate descent with newton direction
                return (-g_j / H_jj);
            }
        }

        arma::rowvec getSubgradients(const arma::rowvec &parameterValues,
                                     const arma::rowvec &gradients,
                                     const tuningParametersEnetGlmnet &tuningParameters)
        {

            arma::rowvec subgradients = gradients;
            double lower, upper;
            int sign;

            for (unsigned int p = 0; p < parameterValues.n_elem; p++)
            {

                // if not regularized: nothing to do here
                if (tuningParameters.weights.at(p) == 0)
                    continue;

                // check if parameter is at non-differentiable place:
                if (parameterValues.at(p) == 0)
                {
                    lower = -tuningParameters.weights.at(p) *
                            tuningParameters.alpha.at(p) *
                            tuningParameters.lambda.at(p);
                    // note: we don't add the ridge part here, because this part is already incorporated
                    // in the differentiable part in gradients
                    upper = -lower;

                    if (lower < gradients.at(p))
                    {
                        subgradients.at(p) = gradients.at(p) + upper;
                        continue;
                    }
                    else if (gradients.at(p) > upper)
                    {
                        subgradients.at(p) = gradients.at(p) + lower;
                        continue;
                    }
                    else
                    {
                        Rcpp::stop("Error in subgradient computation");
                    }
                }
                else
                {
                    // parameter is regularized, but not zeroed
                    sign = (parameterValues.at(p) > 0);
                    if (parameterValues.at(p) < 0)
                        sign = -1;

                    subgradients.at(p) = gradients.at(p) +
                                         sign *
                                             tuningParameters.weights.at(p) *
                                             tuningParameters.alpha.at(p) *
                                             tuningParameters.lambda.at(p);
                }

            } // end for parameter

            return (subgradients);
        }
    };

}
#endif