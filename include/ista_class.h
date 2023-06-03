#ifndef ISTACLASS_H
#define ISTACLASS_H
#include "common_headers.h"

#include "model.h"
#include "fitResults.h"
#include "proximalOperator.h"
#include "penalty.h"
#include "smoothPenalty.h"

// The design follows ensmallen (https://github.com/mlpack/ensmallen) in that the
// user supplies a C++ class with methods fit and gradients which is used
// by the optimizer

// The implementation of ista follows that outlined in
// Beck, A., & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding
// Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences, 2(1),
// 183–202. https://doi.org/10.1137/080716542
// see Remark 3.1 on p. 191 (ISTA with backtracking)

// GIST can be found in
// Gong, P., Zhang, C., Lu, Z., Huang, J., & Ye, J. (2013).
// A General Iterative Shrinkage and Thresholding Algorithm for Non-convex
// Regularized Optimization Problems. Proceedings of the 30th International
// Conference on Machine Learning, 28(2)(2), 37–45.

namespace lessSEM
{

  // convCritInnerIsta
  //
  // Convergence criteria used by the ista optimizer.
  // istaCrit: The approximated fit based on the quadratic approximation
  // h(parameters_k) := fit(parameters_k) +
  // (parameters_k-parameters_kMinus1)*gradients_k^T +
  // (L/2)*(parameters_k-parameters_kMinus1)^2 +
  // penalty(parameters_k)
  // is compared to the exact fit
  //
  // gistCrit:
  // the exact fit is compared to
  // h(parameters_k) := fit(parameters_k) +
  // penalty(parameters_kMinus1) +
  // L*(sigma/2)*(parameters_k-parameters_kMinus1)^2
  enum convCritInnerIsta
  {
    istaCrit,
    gistCrit
  };
  const std::vector<std::string> convCritInnerIsta_txt = {
      "istaCrit",
      "gistCrit"};

  // stepSizeInheritance
  //
  // The ista optimizer provides different rules to be used to find an initial
  // step size. It defines if and how the step size should be carried forward
  // from iteration to iteration.
  // initial: resets the step size to L0 in each iteration
  // istaStepInheritance: takes the previous step size as initial value for the
  // next iteration
  // barzilaiBorwein: uses the Barzilai-Borwein procedure
  // stochasticBarzilaiBorwein: uses the Barzilai-Borwein procedure, but sometimes
  // resets the step size; this can help when the optimizer is caught in a bad spot.
  enum stepSizeInheritance
  {
    initial,
    istaStepInheritance,
    barzilaiBorwein,
    stochasticBarzilaiBorwein
  };
  const std::vector<std::string> stepSizeInheritance_txt = {
      "initial",
      "istaStepInheritance",
      "barzilaiBorwein",
      "stochasticBarzilaiBorwein"};

  // control
  //
  // Settings for the ista optimizer.
  //
  // L0: controls the step size used in the first iteration
  // eta: controls by how much the step size changes in
  // inner iterations with (eta^i)*L, where i is the inner iteration
  // accelerate: if true, the extrapolation parameter is used
  // to accelerate ista (see, e.g., Parikh, N., & Boyd, S. (2013).
  // Proximal Algorithms. Foundations and Trends in Optimization, 1(3), 123–231.,
  // p. 152)
  // maxIterOut: maximal number of outer iterations
  // maxIterIn: maximal number of inner iterations
  // breakOuter: change in fit required to break the outer iteration
  // convCritInner: this is related to the inner breaking condition. ista, as presented by Beck & Teboulle (2009);
  // see Remark 3.1 on p. 191 (ISTA with backtracking) or gist, as presented by Gong et al. (2013) (Equation 3)
  // sigma: sigma in (0,1) is used by the gist convergence criterion. larger
  // sigma enforce larger improvement in fit
  // stepSizeIn: how should step sizes be carried forward? See stepSizeInheritance
  // sampleSize: can be used to scale the fitting function down
  // verbose: if set to a value > 0, the fit every verbose iterations
  // is printed.
  struct control
  {
    double L0;
    double eta;
    bool accelerate;
    int maxIterOut;
    int maxIterIn;
    double breakOuter;
    convCritInnerIsta convCritInner;
    double sigma;
    stepSizeInheritance stepSizeIn;
    int sampleSize;
    int verbose;
  };
  // the name "control" dates back to when there was no glmnet optimizer and is
  // kept for backwards compatability. For consistency, we also define a more
  // explicit name:
  typedef control controlIsta;

  // controlDefault
  //
  // Returns the default setting for the ista optimizer
  // @return object of class control.
  inline control controlDefault()
  {
    control defaultIs = {
        .1,                  // L0
        2,                   // eta
        true,                // accelerate
        1000,                // maxIterOut
        10000,               // maxIterIn
        .00000001,           // breakOuter
        gistCrit,            // convCritInner
        .1,                  // sigma
        istaStepInheritance, // stepSizeInheritance
        1,                   // sample size
        0                    // verbose
    };
    return (defaultIs);
  }

  inline control controlIstaDefault()
  {
    return (controlDefault());
  }

  // We provide two optimizer interfaces: One uses a combination of arma::rowvec and lessSEM::stringVector for starting
  // values and parameter labels respectively. This interface is consistent with the fit and gradient function of the
  // lessSEM::model-class. Alternatively, a numericVector can be passed to the optimizers. This design is rooted in
  // the use of Rcpp::NumericVectors that combine values and labels similar to an R vector. Thus, interfacing to this
  // second function call can be easier when coming from R.

  // ista
  //
  // Implements (variants of) the ista optimizer.
  //
  // @param model_ the model object derived from the model class in model.h
  // @param startingValuesRcpp an Rcpp numeric vector with starting values
  // @parma proximalOperator_ a proximal operator for the penalty function
  // @param
  // @param penalty_ a penalty derived from the penalty class in penalty.h
  // @param smoothPenalty_ a smooth penalty derived from the smoothPenalty class in smoothPenalty.h
  // @param tuningParameters tuning parameters for the penalty function
  // @parma smoothTuningParameters tuning parameters for the smooth penalty function
  // @param control_ settings for the ista optimizer.
  // @return fit result
  template <typename T, typename U> // T is the type of the tuning parameters
  inline lessSEM::fitResults ista(
      model &model_,
      numericVector startingValuesRcpp,
      proximalOperator<T> &proximalOperator_, // proximalOperator takes the tuning parameters
      // as input -> <T>
      penalty<T> &penalty_,             // penalty takes the tuning parameters
      smoothPenalty<U> &smoothPenalty_, // smoothPenalty takes the smooth tuning parameters
      // as input -> <U>
      const T &tuningParameters,       // tuning parameters are of type T
      const U &smoothTuningParameters, // tuning parameters are of type U
      const control &control_ = controlDefault())
  {
    if (control_.verbose != 0)
    {
      print << "Optimizing with ista.\n"
            << "Using "
            << convCritInnerIsta_txt.at(control_.convCritInner)
            << " as inner convergence criterion\n"
            << "Using "
            << stepSizeInheritance_txt.at(control_.stepSizeIn)
            << " as step size inheritance\n"
            << "Tuning parameters: \n eta = "
            << control_.eta
            << "\n"
            << " accelerate = "
            << control_.accelerate
            << "\n"
            << " sigma = "
            << control_.sigma
            << "\n"
            << " breakOuter = "
            << control_.breakOuter
            << std::endl;
    }
    // separate labels and values
    const arma::rowvec startingValues = toArmaVector(startingValuesRcpp);
    const stringVector parameterLabels = startingValuesRcpp.names();

    // prepare parameter vectors
    arma::rowvec parameters_k = startingValues,
                 parameters_kMinus1 = startingValues,
                 parameters_kMinus2 = startingValues,
                 y_k = startingValues; // required for acceleration
    // the following elements will be required to judge the breaking condition
    arma::rowvec parameterChange(startingValues.n_elem);
    arma::rowvec gradientChange(startingValues.n_elem); // necessary for Barzilai Borwein
    arma::mat quadr, parchTimeGrad;
    numericVector randomNumber; // for stochastic Barzilai Borwein

    // prepare fit elements
    double fit_k = (1.0 / control_.sampleSize) * model_.fit(startingValues, parameterLabels) +
                   smoothPenalty_.getValue(parameters_k, parameterLabels, smoothTuningParameters), // ridge penalty part
        fit_kMinus1 = (1.0 / control_.sampleSize) * model_.fit(startingValues, parameterLabels) +
                      smoothPenalty_.getValue(parameters_kMinus1, parameterLabels, smoothTuningParameters), // ridge penalty part,
        penalty_k = 0.0;
    double penalizedFit_k, penalizedFit_kMinus1;
    arma::rowvec gradients_k, gradients_kMinus1, gradient_y_k;

    penalizedFit_k = fit_k +
                     penalty_.getValue(parameters_k, parameterLabels, tuningParameters); // lasso penalty part

    penalizedFit_kMinus1 = fit_kMinus1 +
                           penalty_.getValue(parameters_kMinus1, parameterLabels, tuningParameters); // lasso penalty part

    // the following vector will save the fits of all iterations:
    arma::rowvec fits(control_.maxIterOut + 1);
    fits.fill(arma::datum::nan);
    fits(0) = penalizedFit_kMinus1;

    // prepare gradient elements
    // NOTE: We combine the gradients of the smooth functions (the log-Likelihood)
    // of the model and the smooth penalty function (e.g., ridge)
    gradients_k = (1.0 / control_.sampleSize) * model_.gradients(parameters_k, parameterLabels) +
                  smoothPenalty_.getGradients(parameters_k, parameterLabels, smoothTuningParameters); // ridge part
    gradients_kMinus1 = (1.0 / control_.sampleSize) * model_.gradients(parameters_kMinus1, parameterLabels) +
                        smoothPenalty_.getGradients(parameters_kMinus1, parameterLabels, smoothTuningParameters); // ridge part
    // for acceleration:
    gradient_y_k = (1.0 / control_.sampleSize) * model_.gradients(parameters_kMinus1, parameterLabels) +
                   smoothPenalty_.getGradients(parameters_kMinus1, parameterLabels, smoothTuningParameters); // ridge part

    // breaking flags
    bool breakInner = false, // if true, the inner iteration is exited
        breakOuter = false;  // if true, the outer iteration is exited

    // initialize step size
    double L_kMinus1 = control_.L0, L_k = control_.L0;

    // outer iteration
    for (int outer_iteration = 0; outer_iteration < control_.maxIterOut; outer_iteration++)
    {

      // check if user wants to stop the computation:
#if USE_R
      Rcpp::checkUserInterrupt();
#endif

      for (int inner_iteration = 0; inner_iteration < control_.maxIterIn; inner_iteration++)
      {
        // inner iteration: reduce step size until the convergence criterion is met
        L_k = std::pow(control_.eta, inner_iteration) * L_kMinus1;

        if (control_.accelerate)
        {
          // with acceleration:
          // apply proximal operator to get new parameters for given step size
          // see Parikh, N., & Boyd, S. (2013). Proximal Algorithms. Foundations
          // and Trends in Optimization, 1(3), 123–231. p. 152

          y_k = parameters_kMinus1 +
                (inner_iteration / (inner_iteration + 3)) * (parameters_kMinus1 - parameters_kMinus2);
          gradient_y_k = (1.0 / control_.sampleSize) * model_.gradients(y_k,
                                                                        parameterLabels) +
                         smoothPenalty_.getGradients(y_k,
                                                     parameterLabels,
                                                     smoothTuningParameters);
          parameters_k = proximalOperator_.getParameters(
              y_k,
              gradient_y_k,
              parameterLabels,
              L_k,
              tuningParameters);
        }
        else
        {

          // apply proximal operator to get new parameters for given step size
          parameters_k = proximalOperator_.getParameters(
              parameters_kMinus1,
              gradients_kMinus1,
              parameterLabels,
              L_k,
              tuningParameters);
        }

        // compute new fit; if this fit is non-finite, we can jump to the next
        // iteration
        fit_k = (1.0 / control_.sampleSize) * model_.fit(parameters_k, parameterLabels) +
                smoothPenalty_.getValue(parameters_k, parameterLabels, smoothTuningParameters); // ridge penalty part

        if (!arma::is_finite(fit_k))
          continue;

        // fit_k is only part of the fit we are interested in. We also need
        // the penalty values:
        penalty_k = penalty_.getValue(parameters_k,
                                      parameterLabels,
                                      tuningParameters);

        penalizedFit_k = fit_k +
                         penalty_k; // lasso part

        if (!arma::is_finite(penalizedFit_k))
          continue;

        // to test the convergence criterion, we offer different criteria

        if (control_.convCritInner == istaCrit)
        {
          // ISTA:
          // The approximated fit based on the quadratic approximation
          // h(parameters_k) := fit(parameters_k) +
          // (parameters_k-parameters_kMinus1)*gradients_k^T +
          // (L/2)*(parameters_k-parameters_kMinus1)^2 +
          // penalty(parameters_k)
          // is compared to the exact fit
          parameterChange = parameters_k - parameters_kMinus1;
          quadr = parameterChange * arma::trans(parameterChange);           // always positive
          parchTimeGrad = parameterChange * arma::trans(gradients_kMinus1); // can be
          // positive or negative

          breakInner = penalizedFit_k <= (fit_kMinus1 +
                                          parchTimeGrad(0, 0) +
                                          (L_k / 2.0) * quadr(0, 0) +
                                          penalty_k);
        }
        else if (control_.convCritInner == gistCrit)
        {

          // GIST:
          // the exact fit is compared to
          // h(parameters_k) := fit(parameters_k) +
          // penalty(parameters_kMinus1) +
          // L*(sigma/2)*(parameters_k-parameters_kMinus1)^2
          //
          parameterChange = parameters_k - parameters_kMinus1;
          quadr = parameterChange * arma::trans(parameterChange); // always positive

          breakInner = penalizedFit_k <= (penalizedFit_kMinus1 -
                                          L_k * (control_.sigma / 2.0) * quadr(0, 0));
        }

        if (breakInner)
        {
          // compute gradients at new position
          gradients_k = (1.0 / control_.sampleSize) * model_.gradients(parameters_k,
                                                                       parameterLabels) +
                        smoothPenalty_.getGradients(parameters_k,
                                                    parameterLabels,
                                                    smoothTuningParameters); // ridge part

          // if any of the gradients is non-finite, we can skip to a
          // smaller step size
          if (!arma::is_finite(gradients_k))
            continue;

          // if everything worked out fine, we break the inner iteration
          break;

        } // end break inner
      }   // end inner iteration

      // print fit info
      if ((control_.verbose > 0) && (outer_iteration % control_.verbose == 0))
      {
        print << "Fit in iteration outer_iteration " << outer_iteration + 1 << ": " << penalizedFit_k << " (" << fit_k << " + " << penalty_k << ")" << std::endl;
        print << parameters_k << std::endl;
      }

      if ((!breakInner) && (control_.verbose < 0))
      {
        warn("Inner iterations did not improve the fit --> resetting L.");
        L_kMinus1 = control_.L0;
        continue;
      }

      gradients_k = (1.0 / control_.sampleSize) * model_.gradients(parameters_k,
                                                                   parameterLabels) +
                    smoothPenalty_.getGradients(parameters_k,
                                                parameterLabels,
                                                smoothTuningParameters); // ridge part

      fits(outer_iteration + 1) = penalizedFit_k;

      // check outer breaking condition
      breakOuter = std::abs(fits(outer_iteration + 1) - fits(outer_iteration)) < control_.breakOuter;

      if (breakOuter)
      {
        break;
      }

      // define new initial step size
      if (control_.stepSizeIn == initial)
      {

        L_kMinus1 = control_.L0;
      }
      else if (control_.stepSizeIn == barzilaiBorwein ||
               control_.stepSizeIn == stochasticBarzilaiBorwein)
      {

        parameterChange = parameters_k - parameters_kMinus1;
        gradientChange = gradients_k - gradients_kMinus1;

        quadr = (parameterChange)*arma::trans(parameterChange);
        parchTimeGrad = parameterChange * arma::trans(gradientChange);

        L_kMinus1 = parchTimeGrad(0, 0) / quadr(0, 0);

        if (L_kMinus1 < 1e-10 || L_kMinus1 > 1e10)
          L_kMinus1 = control_.L0;

        randomNumber = unif(1, 0.0, 1.0);
        if ((control_.stepSizeIn == stochasticBarzilaiBorwein) &&
            (randomNumber.at(0) < 0.25))
        {
          L_kMinus1 = control_.L0; // reset with 25% probability
        }
      }
      else if (control_.stepSizeIn == istaStepInheritance)
      {

        L_kMinus1 = L_k;
      }
      else
      {
        error("Unknown step inheritance.");
      }

      // for next iteration: save current values as previous values
      fit_kMinus1 = fit_k;
      penalizedFit_kMinus1 = penalizedFit_k;
      parameters_kMinus2 = parameters_kMinus1;
      parameters_kMinus1 = parameters_k;
      gradients_kMinus1 = gradients_k;
    }

    fitResults fitResults_;

    fitResults_.convergence = breakOuter;
    fitResults_.fit = control_.sampleSize * penalizedFit_k; // rescale for -2log-Likelihood
    fitResults_.fits = control_.sampleSize * fits;          // rescale for -2log-Likelihood
    fitResults_.parameterValues = parameters_k;

    return (fitResults_);
  }

  // ista
  //
  // Implements (variants of) the ista optimizer.
  //
  // @param model_ the model object derived from the model class in model.h
  // @param startingValues an arma::rowvec numeric vector with starting values
  // @param parameterLabels a lessSEM::stringVector with labels for parameters
  // @parma proximalOperator_ a proximal operator for the penalty function
  // @param
  // @param penalty_ a penalty derived from the penalty class in penalty.h
  // @param smoothPenalty_ a smooth penalty derived from the smoothPenalty class in smoothPenalty.h
  // @param tuningParameters tuning parameters for the penalty function
  // @parma smoothTuningParameters tuning parameters for the smooth penalty function
  // @param control_ settings for the ista optimizer.
  // @return fit result
  template <typename T, typename U> // T is the type of the tuning parameters
  inline lessSEM::fitResults ista(
      model &model_,
      arma::rowvec startingValues,
      stringVector parameterLabels,
      proximalOperator<T> &proximalOperator_, // proximalOperator takes the tuning parameters
      // as input -> <T>
      penalty<T> &penalty_,             // penalty takes the tuning parameters
      smoothPenalty<U> &smoothPenalty_, // smoothPenalty takes the smooth tuning parameters
      // as input -> <U>
      const T &tuningParameters,       // tuning parameters are of type T
      const U &smoothTuningParameters, // tuning parameters are of type U
      const control &control_ = controlDefault())
  {
    numericVector startingValuesNumVec = toNumericVector(startingValues);
    startingValuesNumVec.names() = parameterLabels;

    return (
        ista(
            model_,
            startingValuesNumVec,
            proximalOperator_, // proximalOperator takes the tuning parameters
            // as input -> <T>
            penalty_,       // penalty takes the tuning parameters
            smoothPenalty_, // smoothPenalty takes the smooth tuning parameters
            // as input -> <U>
            tuningParameters,       // tuning parameters are of type T
            smoothTuningParameters, // tuning parameters are of type U
            control_));
  }

} // end namespace

#endif