#ifndef GLMNETCLASS_H
#define GLMNETCLASS_H
#include "common_headers.h"

#include "model.h"
#include "fitResults.h"
#include "glmnet_lasso.h"
#include "glmnet_ridge.h"
#include "enet.h"
#include "bfgs.h"

// The design follows ensmallen (https://github.com/mlpack/ensmallen) in that the
// user supplies a C++ class with methods fit and gradients which is used
// by the optimizer

// The implementation of GLMNET follows that outlined in
// 1) Friedman, J., Hastie, T., & Tibshirani, R. (2010).
// Regularization Paths for Generalized Linear Models via Coordinate Descent.
// Journal of Statistical Software, 33(1), 1–20. https://doi.org/10.18637/jss.v033.i01
// 2) Yuan, G.-X., Chang, K.-W., Hsieh, C.-J., & Lin, C.-J. (2010).
// A Comparison of Optimization Methods and Software for Large-scale
// L1-regularized Linear Classification. Journal of Machine Learning Research, 11, 3183–3234.
// 3) Yuan, G.-X., Ho, C.-H., & Lin, C.-J. (2012).
// An improved GLMNET for l1-regularized logistic regression.
// The Journal of Machine Learning Research, 13, 1999–2030. https://doi.org/10.1145/2020408.2020421

namespace lessSEM
{

  // convergenceCriteriaGlmnet
  //
  // Specifies the convergence criteria that are currently available for the glmnet optimizer.
  // The optimization stops if the specified convergence criterion is met.
  //
  // GLMNET: Uses the convergence criterion outlined in Yuan et al. (2012) for GLMNET.
  // fitChange: Uses the change in fit from one iteration to the next.
  // gradients: Uses the gradients; if all are (close to) zero, the minimum is found
  enum convergenceCriteriaGlmnet
  {
    GLMNET,
    fitChange,
    gradients
  };
  const std::vector<std::string> convergenceCriteriaGlmnet_txt = {
      "GLMNET",
      "fitChange",
      "gradients"};

  // controlGLMNET
  //
  // Allows you to adapt the optimizer settings for the glmnet optimizer
  //
  // @param initialHessian initial Hessian matrix fo the optimizer.
  // @param stepSize Initial stepSize of the outer iteration (theta_{k+1} = theta_k + stepSize * Stepdirection)
  // @param sigma only relevant when lineSearch = 'GLMNET'. Controls the sigma parameter in Yuan, G.-X., Ho, C.-H., & Lin, C.-J. (2012). An improved GLMNET for l1-regularized logistic regression. The Journal of Machine Learning Research, 13, 1999–2030. https://doi.org/10.1145/2020408.2020421.
  // @param gamma Controls the gamma parameter in Yuan, G.-X., Ho, C.-H., & Lin, C.-J. (2012). An improved GLMNET for l1-regularized logistic regression. The Journal of Machine Learning Research, 13, 1999–2030. https://doi.org/10.1145/2020408.2020421. Defaults to 0.
  // @param maxIterOut Maximal number of outer iterations
  // @param maxIterIn Maximal number of inner iterations
  // @param maxIterLine Maximal number of iterations for the line search procedure
  // @param breakOuter Stopping criterion for outer iterations
  // @param breakInner Stopping criterion for inner iterations
  // @param convergenceCriterion which convergence criterion should be used for the outer iterations? possible are 0 = GLMNET, 1 = fitChange, 2 = gradients.
  // Note that in case of gradients and GLMNET, we divide the gradients (and the Hessian) of the log-Likelihood by N as it would otherwise be
  // considerably more difficult for larger sample sizes to reach the convergence criteria.
  // @param verbose 0 prints no additional information, > 0 prints GLMNET iterations
  struct controlGLMNET
  {
    arma::mat initialHessian;
    double stepSize;
    double sigma;
    double gamma;
    int maxIterOut; // maximal number of outer iterations
    int maxIterIn;  // maximal number of inner iterations
    int maxIterLine;
    double breakOuter; // change in fit required to break the outer iteration
    double breakInner;
    convergenceCriteriaGlmnet convergenceCriterion; // this is related to the inner
    // breaking condition.
    int verbose; // if set to a value > 0, the fit every verbose iterations
    // is printed.
  };

  // controlGlmnetDefault
  //
  // Returns the default settings for the glmnet optimizer.
  // @return object of struct controlGLMNET.
  inline controlGLMNET controlGlmnetDefault()
  {
    arma::mat initialHessian(1, 1);
    initialHessian.fill(1.0);
    controlGLMNET defaultIs = {
        initialHessian, // initial hessian will be diagonal with 100 as diagonal values
        .9,             // stepSize;
        1e-5,           // sigma
        0,              // gamma
        1000,           // maxIterOut; // maximal number of outer iterations
        1000,           // maxIterIn; // maximal number of inner iterations
        500,            // maxIterLine;
        1e-8,           // breakOuter; // change in fit required to break the outer iteration
        1e-10,          // breakInner;
        fitChange,      // convergenceCriterion; // this is related to the inner
        // breaking condition.
        0 // verbose; // if set to a value > 0, the fit every verbose iterations
          // is printed.
    };
    return (defaultIs);
  }

  // glmnetInner
  //
  // The glmnet optimizer has an outer and an inner optimization loop. This function implements
  // the inner optimization loop which returns the step direction.
  // To this end, the function q_k(direction) = direction * gradients_kMinus1 + .5*direction*Hessian_kMinus1 * direction + sum_j(lambda_j*alpha_j*|parameters_kMinus1_j + direction_j| - lambda_j*alpha_j*|parameters_kMinus1_j|) is minimized.
  // @param parameters_kMinus1 parameter estimates from previous iteration k-1
  // @param gradients_kMinus1 gradients from previous iteration
  // @param Hessian Hessian_kMinus1 Hessian from previous iteration
  // @param lambda tuning parameter lambda
  // @param alpha tuning parameter alpha
  // @param weights tuning parameter weights
  // @param maxIterIn Maximal number of inner iterations
  // @param breakInner Stopping criterion for inner iterations
  // @param verbose 0 prints no additional information, > 0 prints GLMNET iterations
  template <typename nonsmoothPenalty,
            typename tuning>
  inline arma::rowvec glmnetInner(const arma::rowvec &parameters_kMinus1,
                                  const arma::rowvec &gradients_kMinus1,
                                  const arma::mat &Hessian,
                                  nonsmoothPenalty &penalty_,
                                  const tuning &tuningParameters,
                                  const int maxIterIn,
                                  const double breakInner,
                                  const int verbose)
  {
    arma::rowvec stepDirection = parameters_kMinus1;
    stepDirection.fill(0.0);
    arma::rowvec z = parameters_kMinus1;
    z.fill(0.0);
    //arma::rowvec parameters_k = parameters_kMinus1;
    //parameters_k.fill(0.0);
    arma::colvec HessTimesZ(Hessian.n_rows, arma::fill::zeros);
    arma::mat HessDiag(Hessian.n_rows, Hessian.n_cols, arma::fill::zeros);//,
        //zChange(1, 1, arma::fill::zeros);
    double z_j;

    HessDiag.diag() = Hessian.diag();

    // the order in which parameters are updated should be random
    numericVector randOrder(stepDirection.n_elem);
    numericVector sampleFrom(stepDirection.n_elem);
    for (unsigned int i = 0; i < stepDirection.n_elem; i++)
      sampleFrom.at(i) = i;

    for (int it = 0; it < maxIterIn; it++)
    {

      // reset direction z
      z.fill(arma::fill::zeros);
      // z_old.fill(arma::fill::zeros);

      // iterate over parameters in random order
      randOrder = sample(sampleFrom, stepDirection.n_elem, false);

      for (unsigned int p = 0; p < stepDirection.n_elem; p++)
      {
        // get the update to the parameter:
        z_j = penalty_.getZ(
            randOrder.at(p),
            parameters_kMinus1,
            gradients_kMinus1,
            stepDirection,
            Hessian,
            tuningParameters);
        z.col(randOrder.at(p)) = z_j;
        stepDirection.col(randOrder.at(p)) += z_j;
      }

      // check inner stopping criterion:
      HessTimesZ = HessDiag * arma::pow(arma::trans(z), 2);

      // zChange = z*arma::trans(z_old);
      // if(useMultipleConvergenceCriteria & (zChange(0,0) < breakInner)){
      //   break;
      // }

      if (HessTimesZ.max() < breakInner)
      {
        break;
      }
      // z_old = z;
    }

    return (stepDirection);
  }

  // glmnetLineSearch
  //
  // Given a step direction "direction", the line search procedure will find an adequate
  // step length s in this direction. The new parameter values are then given by
  // parameters_k = parameters_kMinus1 + s*direction
  // @param model_ the model object derived from the model class in model.h
  // @param penalty_ a penalty derived from the penalty class in penalty.h
  // @param smoothPenalty a smooth penalty derived from the smoothPenalty class in smoothPenalty.h
  // @param parameters_kMinus1 parameter estimates from previous iteration k-1
  // @param parameterLabels names of the parameters
  // @param direction step direction
  // @param fit_kMinus1 fit from previous iteration
  // @param gradients_kMinus1 gradients from previous iteration
  // @param Hessian_kMinus1 Hessian from previous iteration
  // @param tuningParameters tuning parameters for the penalty function
  // @param stepSize Initial stepSize of the outer iteration (theta_{k+1} = theta_k + stepSize * Stepdirection)
  // @param sigma only relevant when lineSearch = 'GLMNET'. Controls the sigma parameter in Yuan, G.-X., Ho, C.-H., & Lin, C.-J. (2012). An improved GLMNET for l1-regularized logistic regression. The Journal of Machine Learning Research, 13, 1999–2030. https://doi.org/10.1145/2020408.2020421.
  // @param gamma Controls the gamma parameter in Yuan, G.-X., Ho, C.-H., & Lin, C.-J. (2012). An improved GLMNET for l1-regularized logistic regression. The Journal of Machine Learning Research, 13, 1999–2030. https://doi.org/10.1145/2020408.2020421. Defaults to 0.
  // @param maxIterLine Maximal number of iterations for the line search procedure
  // @param verbose 0 prints no additional information, > 0 prints GLMNET iterations
  // @return vector with updated parameters (parameters_k)
  template <typename nonsmoothPenalty, typename smoothPenalty,
            typename tuning>
  inline arma::rowvec glmnetLineSearch(
      model &model_,
      nonsmoothPenalty &penalty_,
      smoothPenalty &smoothPenalty_,
      const arma::rowvec &parameters_kMinus1,
      const stringVector &parameterLabels,
      const arma::rowvec &direction,
      const double fit_kMinus1,
      const arma::rowvec &gradients_kMinus1,
      const arma::mat &Hessian_kMinus1,

      const tuning &tuningParameters,

      const double stepSize,
      const double sigma,
      const double gamma,
      const int maxIterLine,
      const int verbose)
  {

    arma::rowvec gradients_k(gradients_kMinus1.n_rows);
    gradients_k.fill(arma::datum::nan);
    arma::rowvec parameters_k(gradients_kMinus1.n_rows);
    parameters_k.fill(arma::datum::nan);
    numericVector randomNumber;

    double fit_k; // new fit value of differentiable part
    double p_k;   // new penalty value
    double f_k;   // new combined fit

    // get penalized M2LL for step size 0:

    double pen_0 = penalty_.getValue(parameters_kMinus1,
                                     parameterLabels,
                                     tuningParameters);
    // Note: we see the smooth penalty as part of the smooth
    // objective function and not as part of the non-differentiable
    // penalty
    double f_0 = fit_kMinus1 + pen_0;
    // needed for convergence criterion (see Yuan et al. (2012), Eq. 20)
    double pen_d = penalty_.getValue(parameters_kMinus1 + direction,
                                     parameterLabels,
                                     tuningParameters);

    double currentStepSize;
    // a step size of >= 1 would result in no change or in an increasing step
    // size
    if (stepSize >= 1)
    {
      currentStepSize = .9;
    }
    else
    {
      currentStepSize = stepSize;
    }

    randomNumber = unif(1, 0.0, 1.0);
    if (randomNumber.at(0) < 0.25)
    {
      numericVector tmp = unif(1, .5, .99);
      currentStepSize = tmp.at(0);
    }

    bool converged = false;

    for (int iteration = 0; iteration < maxIterLine; iteration++)
    {

      // set step size
      currentStepSize = std::pow(stepSize, iteration); // starts with 1 and
      // then decreases with each iteration

      parameters_k = parameters_kMinus1 + currentStepSize * direction;

      fit_k = model_.fit(parameters_k,
                         parameterLabels) +
              smoothPenalty_.getValue(parameters_k,
                                      parameterLabels,
                                      tuningParameters);

      if (!arma::is_finite(fit_k))
      {
        // skip to next iteration and try a smaller step size
        continue;
      }

      // compute g(stepSize) = g(x+td) + p(x+td) - g(x) - p(x),
      // where g is the differentiable part and p the non-differentiable part
      // p(x+td):
      p_k = penalty_.getValue(parameters_k,
                              parameterLabels,
                              tuningParameters);

      // g(x+td) + p(x+td)
      f_k = fit_k + p_k;

      // test line search criterion. g(stepSize) must show a large enough decrease
      // to be accepted
      // see Equation 20 in Yuan, G.-X., Ho, C.-H., & Lin, C.-J. (2012).
      // An improved GLMNET for l1-regularized logistic regression.
      // The Journal of Machine Learning Research, 13, 1999–2030.
      // https://doi.org/10.1145/2020408.2020421

      arma::mat compareTo =
          gradients_kMinus1 * arma::trans(direction) + // gradients and direction typically show
          // in the same direction -> positive
          gamma * (direction * Hessian_kMinus1 * arma::trans(direction)) + // always positive
          pen_d - pen_0;
      // gamma is set to zero by Yuan et al. (2012)
      // if sigma is 0, no decrease is necessary

      converged = f_k - f_0 <= sigma * currentStepSize * compareTo(0, 0);

      if (converged)
      {
        // check if gradients can be computed at the new location;
        // this can often cause issues
        gradients_k = model_.gradients(parameters_k,
                                       parameterLabels);

        if (!arma::is_finite(gradients_k))
        {
          // go to next iteration and test smaller step size
          continue;
        }
        // else
        break;
      }

    } // end line search

    return (parameters_k);
  }

  // glmnet
  //
  // Optimize a model using the glmnet procedure.
  // @param model_ the model object derived from the model class in model.h
  // @param startingValuesRcpp an Rcpp numeric vector with starting values
  // @param penalty_ a penalty derived from the penalty class in penalty.h
  // @param smoothPenalty_ a smooth penalty derived from the smoothPenalty class in smoothPenalty.h
  // @param tuningParameters tuning parameters for the penalty functions. Note that both penalty functions must
  // take the same tuning parameters.
  // @param control_ settings for the glmnet optimizer.
  // @return fit result
  template <typename nonsmoothPenalty, typename smoothPenalty,
            typename tuning>
  inline lessSEM::fitResults glmnet(model &model_,
                                    numericVector startingValuesRcpp,
                                    nonsmoothPenalty &penalty_,
                                    smoothPenalty &smoothPenalty_,
                                    const tuning &tuningParameters,
                                    const controlGLMNET &control_ = controlGlmnetDefault())
  {

    if (control_.verbose != 0)
    {
      print << "Optimizing with glmnet.\n";
    }

    // separate labels and values
    arma::rowvec startingValues = toArmaVector(startingValuesRcpp);
    stringVector parameterLabels = startingValuesRcpp.names();

    // prepare parameter vectors
    arma::rowvec parameters_k = startingValues,
                 parameters_kMinus1 = startingValues;
    arma::rowvec direction(startingValues.n_elem);

    // prepare fit elements
    // fit of the smooth part of the fit function
    double fit_k = model_.fit(parameters_k,
                              parameterLabels) +
                   smoothPenalty_.getValue(parameters_k,
                                           parameterLabels,
                                           tuningParameters);
    double fit_kMinus1 = model_.fit(parameters_kMinus1,
                                    parameterLabels) +
                         smoothPenalty_.getValue(parameters_kMinus1,
                                                 parameterLabels,
                                                 tuningParameters);
    // add non-differentiable part
    double penalizedFit_k = fit_k +
                            penalty_.getValue(parameters_k,
                                              parameterLabels,
                                              tuningParameters);

    double penalizedFit_kMinus1 = fit_kMinus1 +
                                  penalty_.getValue(parameters_kMinus1,
                                                    parameterLabels,
                                                    tuningParameters);

    // the following vector will save the fits of all iterations:
    arma::rowvec fits(control_.maxIterOut + 1);
    fits.fill(NA_REAL);
    fits(0) = penalizedFit_kMinus1;

    // prepare gradient elements
    // NOTE: We combine the gradients of the smooth functions (the log-Likelihood)
    // of the model and the smooth penalty function (e.g., ridge)
    arma::rowvec gradients_k = model_.gradients(parameters_k,
                                                parameterLabels) +
                               smoothPenalty_.getGradients(parameters_k,
                                                           parameterLabels,
                                                           tuningParameters); // ridge part
    arma::rowvec gradients_kMinus1 = model_.gradients(parameters_kMinus1,
                                                      parameterLabels) +
                                     smoothPenalty_.getGradients(parameters_kMinus1,
                                                                 parameterLabels,
                                                                 tuningParameters); // ridge part

    // prepare Hessian elements
    arma::mat Hessian_k(startingValues.n_elem, startingValues.n_elem, arma::fill::zeros),
        Hessian_kMinus1(startingValues.n_elem, startingValues.n_elem, arma::fill::zeros);
    if ((control_.initialHessian.n_cols == 1) && (control_.initialHessian.n_rows == 1))
    {
      // Hessian comes from default initializer and has to be redefined
      double hessianValue = control_.initialHessian(0, 0);
      Hessian_k.diag().fill(hessianValue);
      Hessian_kMinus1.diag().fill(hessianValue);
    }
    else
    {
      Hessian_k = control_.initialHessian;
      Hessian_kMinus1 = control_.initialHessian;
    }

    // breaking flags
    bool breakOuter = false; // if true, the outer iteration is exited

    // outer iteration
    for (int outer_iteration = 0; outer_iteration < control_.maxIterOut; outer_iteration++)
    {

      // check if user wants to stop the computation:
#if USE_R
      Rcpp::checkUserInterrupt();
#endif

      // the gradients will be used by the inner iteration to compute the new
      // parameters
      gradients_kMinus1 = model_.gradients(parameters_kMinus1, parameterLabels) +
                          smoothPenalty_.getGradients(parameters_kMinus1, parameterLabels, tuningParameters); // ridge part

      // find step direction
      direction = glmnetInner(parameters_kMinus1,
                              gradients_kMinus1,
                              Hessian_kMinus1,
                              penalty_,
                              tuningParameters,
                              control_.maxIterIn,
                              control_.breakInner,
                              control_.verbose);

      // find length of step in direction
      parameters_k = glmnetLineSearch(model_,
                                      penalty_,
                                      smoothPenalty_,
                                      parameters_kMinus1,
                                      parameterLabels,
                                      direction,
                                      fit_kMinus1,
                                      gradients_kMinus1,
                                      Hessian_kMinus1,

                                      tuningParameters,

                                      control_.stepSize,
                                      control_.sigma,
                                      control_.gamma,
                                      control_.maxIterLine,
                                      control_.verbose);

      // get gradients of differentiable part
      gradients_k = model_.gradients(parameters_k,
                                     parameterLabels) +
                    smoothPenalty_.getGradients(parameters_k,
                                                parameterLabels,
                                                tuningParameters);
      // fit of the smooth part of the fit function
      fit_k = model_.fit(parameters_k,
                         parameterLabels) +
              smoothPenalty_.getValue(parameters_k,
                                      parameterLabels,
                                      tuningParameters);
      // add non-differentiable part
      penalizedFit_k = fit_k +
                       penalty_.getValue(parameters_k,
                                         parameterLabels,
                                         tuningParameters);

      fits(outer_iteration + 1) = penalizedFit_k;

      // print fit info
      if (control_.verbose > 0 && outer_iteration % control_.verbose == 0)
      {
        print << "Fit in iteration outer_iteration "
              << outer_iteration + 1
              << ": "
              << penalizedFit_k
              << "\n"
              << parameters_k
              << "\n";
      }

      // Approximate Hessian using BFGS
      Hessian_k = lessSEM::BFGS(
          parameters_kMinus1,
          gradients_kMinus1,
          Hessian_kMinus1,
          parameters_k,
          gradients_k,
          true,
          .001,
          control_.verbose == -99);

      // check convergence
      if (control_.convergenceCriterion == GLMNET)
      {
        arma::mat HessDiag = arma::eye(Hessian_k.n_rows,
                                       Hessian_k.n_cols);
        HessDiag.fill(0.0);
        HessDiag.diag() = Hessian_k.diag();
        try
        {
          breakOuter = max(HessDiag * arma::pow(arma::trans(direction), 2)) < control_.breakOuter;
        }
        catch (...)
        {
          error("Error while computing convergence criterion");
        }
      }
      if (control_.convergenceCriterion == fitChange)
      {
        try
        {
          breakOuter = std::abs(fits(outer_iteration + 1) -
                                fits(outer_iteration)) <
                       control_.breakOuter;
        }
        catch (...)
        {
          error("Error while computing convergence criterion");
        }
      }
      if (control_.convergenceCriterion == gradients)
      {
        try
        {
          arma::rowvec subGradients = penalty_.getSubgradients(
              parameters_k,
              gradients_k,
              tuningParameters);

          // check if all gradients are below the convergence criterion:
          breakOuter = arma::sum(arma::abs(subGradients) < control_.breakOuter) ==
                       subGradients.n_elem;
        }
        catch (...)
        {
          error("Error while computing convergence criterion");
        }
      }

      if (breakOuter)
      {
        break;
      }

      // for next iteration: save current values as previous values
      fit_kMinus1 = fit_k;
      penalizedFit_kMinus1 = penalizedFit_k;
      parameters_kMinus1 = parameters_k;
      gradients_kMinus1 = gradients_k;
      Hessian_kMinus1 = Hessian_k;

    } // end outer iteration

    if (!breakOuter)
    {
      warn("Outer iterations did not converge");
    }

    fitResults fitResults_;

    fitResults_.convergence = breakOuter;
    fitResults_.fit = penalizedFit_k;
    fitResults_.fits = fits;
    fitResults_.parameterValues = parameters_k;
    fitResults_.Hessian = Hessian_k;

    return (fitResults_);

  } // end glmnet

} // end namespace

#endif
