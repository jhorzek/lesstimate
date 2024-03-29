# The Optimizer Design

Note: We will not cover the optimizers themselves; here, we recommend
reading the following articles:

**GLMNET**:

* Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of Statistical Software, 33(1), 1–20. https://doi.org/10.18637/jss.v033.i01
* Yuan, G.-X., Ho, C.-H., & Lin, C.-J. (2012). An improved GLMNET for l1-regularized logistic regression. The Journal of Machine Learning Research, 13, 1999–2030. https://doi.org/10.1145/2020408.2020421

**ISTA**:

* Beck, A., & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences, 2(1), 183–202. https://doi.org/10.1137/080716542
* Gong, P., Zhang, C., Lu, Z., Huang, J., & Ye, J. (2013). A general iterative shrinkage and thresholding algorithm for non-convex regularized optimization problems. Proceedings of the 30th International Conference on Machine Learning, 28(2)(2), 37–45.
* Parikh, N., & Boyd, S. (2013). Proximal Algorithms. Foundations and Trends in Optimization, 1(3), 123–231.


## The Fitting Function

The objective of the optimizers is to minimize the fitting function. In both,
glmnet and ista we assume that this fitting function is given by a differentiable
part and a non-differentiable part. To be more specific, the fitting function
is given by:

$$f(\pmb{\theta}) = l(\pmb\theta) + s(\pmb\theta,\pmb{t}_s) + p(\pmb\theta,\pmb{t}_p)$$

where $l(\pmb\theta)$ is the unregularized fitting function 
(e.g., the -2log-likelihood) in the SEMs implemented in **lessSEM**. 
$s(\pmb\theta,\pmb{t}_s)$ is a differentiable (smooth) penalty function 
(e.g., a ridge penalty) and $p(\pmb\theta,\pmb{t}_p)$ is a non-differentiable
penalty function (e.g., a lasso penalty). All three functions take the parameter
estimates $\pmb\theta$ as input and return a single value as output. The penalty 
functions $s(\pmb\theta,\pmb{t}_s)$ and $p(\pmb\theta,\pmb{t}_p)$ additionally
expect vectors with tuning parameters--$\pmb{t}_s$ in case of the smooth penalty
and $\pmb{t}_p$ in case of the non-differentiable penalty. Thus, in theory both
penalty functions can use different tuning parameters. 

A prototypical example for fitting functions of the form defined above is the 
elastic net. Here, 

$$f(\pmb{\theta}) = l(\pmb\theta) + (1-\alpha)\lambda \sum_j\theta_j^2 + \alpha\lambda\sum_j| \theta_j|$$

The elastic net is a combination of a ridge penalty $\lambda \sum_j\theta_j^2$ and a lasso penalty $\lambda\sum_j| \theta_j|$. Note
that in this case, both penalties take in the same tuning parameters 
($\lambda$ and $\alpha$).

With this in mind, we can now have a closer look at the optimization functions.
We will start with glmnet (see function glmnet in the file inst/include/glmnet_class.hpp).
This function is called as follows:



```
inline less::fitResults glmnet(model& model_, 
                                  numericVector startingValuesRcpp,
                                  penaltyLASSOGlmnet& penalty_,
                                  penaltyRidgeGlmnet& smoothPenalty_, 
                                  const tuningParametersEnetGlmnet tuningParameters,
                                  const controlGLMNET& control_ = controlGlmnetDefault())
{...}
```

The first argument is a model. This model has to be created by you and must 
inherit from the less::model class (see lessLM for an example). Most
importantly, this model must provide means to get the value of the first 
part of the fitting function: $l(\pmb\theta)$. It must also provide means
to compute the gradients of your fitting function.

The next argument are the starting values which are given as an numericVector. If you are using R, this is an Rcpp::NumericVector, in C++ a less::numericVector. This type is chosen because it can have labels 
and in its current implementation **lesstimate** expects that you give 
all your startingValues names.

Now come the actual penalty functions. The first one is the non-differentiable
penalty: the lasso $p(\pmb\theta)$. This must be an object of class penaltyLASSOGlmnet
which can be created with `less::penaltyLASSOGlmnet()`'. Next comes the differentiable
ridge penalty which must be of class `penaltyRidgeGlmnet` and can be created with 
`less::penaltyRidgeGlmnet`. 

Now, the tuning parameters deviate a bit from what we discussed before. We said
that the differentiable and the non-differentiable penalty functions will each
take their own vector of tuning parameters ($\pmb t_s$ and $\pmb t_p$ respectively).
Note, however, that the elastic net uses the same tuning parameters for both,
ridge and lasso penalty. In the glmnet optimizer we therefore decided to combine
both of these into one: the `tuningParametersEnetGlmnet` struct. `tuningParametersEnetGlmnet` 
has three slots: alpha (to set $\alpha$), lambda (to set $\lambda$) and
a slot called weights. Now, the weights allow us to switch off the penalty for
selected parameters. For instance, in a linear regression we would not want to 
penalize the intercept. To this end, the fitting function that is actually
implemented internally is given by

$$f(\pmb{\theta}) = l(\pmb\theta) + (1-\alpha)\lambda \sum_j\omega_j \theta_j^2 + \alpha\lambda\sum_j\omega_j| \theta_j|$$

If we set $\omega_j = 0$ for a specific parameter, this parameter is unregularized.
Setting $\omega_j = 1$ means that parameter $j$ is penalized. $\omega_j$ can
also take any other value (e.g., $\omega_j = .4123$) which allows for penalties
such as the adaptive lasso. Importantly, the weights vector must be of the 
same length as `startingValuesRcpp`. That is, each parameter must have a weight
associated with it in the weights vector.

Finally, there is the `controlGLMNET` argument. This argument lets us fine tune the 
optimizer. To use the control argument, create a new control object in C++ as
follows:


```
less::controlGLMNET myControlObject = less::controlGlmnetDefault();
```

Now, you can tweak each element of myControlObject; e.g.,


```
myControlObject.maxIterOut = 100 // only 100 outer iterations
```


If you take a closer look at how the two penalty functions are handled within glmnet,
you will realize that we basically absorb the differentiable penalty in the 
normal fitting function. That is, only the non-differentiable part gets a special
treatment, while the differentiable part is simply added to the differntiable
$l(\pmb\theta)$. To give an example:


```
gradients_kMinus1 = model_.gradients(parameters_kMinus1, parameterLabels) +
      smoothPenalty_.getGradients(parameters_kMinus1, parameterLabels, tuningParameters); // ridge part
```

Note how the gradients of $l(\pmb\theta)$ and $s(\pmb\theta,\pmb{t}_s)$ are combined into one.

## The ista variants

Besides the glmnet optimizer, we also implemented variants of ista. These are
based on the publications mentioned above. The fitting function is again given
by 

$$f(\pmb{\theta}) = l(\pmb\theta) + s(\pmb\theta,\pmb{t}_s) + p(\pmb\theta,\pmb{t}_p)$$

In the 
following, we will build a lot on what we've already discussed regarding the 
glmnet optimizer above. 

First, let's have a look at the ista function;


```
template<typename T, typename U> // T is the type of the tuning parameters
inline less::fitResults ista(
    model& model_, 
    numericVector startingValuesRcpp,
    proximalOperator<T>& proximalOperator_, // proximalOperator takes the tuning parameters
    // as input -> <T>
    penalty<T>& penalty_, // penalty takes the tuning parameters
    smoothPenalty<U>& smoothPenalty_, // smoothPenalty takes the smooth tuning parameters
    // as input -> <U>
    const T& tuningParameters, // tuning parameters are of type T
    const U& smoothTuningParameters, // tuning parameters are of type U
    const control& control_ = controlDefault()
)
{...}
```

This function is more complicated that the glmnet function discussed above. 
But, let's start with the part that stays the same: First, we still have
to pass our model to the function. This model must have a fit and a gradients function
which return the fit and the gradient respectively. Next, the function again expects
us to provide starting values as an numericVector. We will skip the 
`proximalOperator` and the `penalty` for the moment (these relate to $p(\pmb\theta,\pmb t_p)$)
and concentrate on the `smoothPenalty` first. This is the function $s(\pmb\theta,\pmb t_s)$. In our previous example, we looked at the elastic net
penalty, where the smooth penalty is a ridge penalty function. Now, in the ista
optimizer, we can also pass in the ridge penalty as a smooth penalty. In fact, 
this is exactly what we do when we use ista to fit the elastic net. 
This smooth penalty has the tuning
parameters $\pmb t_s$ which are called `smoothTuningParameters` in the function call.
In case of the elastic net, these would again be $\alpha$ and $\lambda$ (and the
weights vector). Similar to the glmnet procedure outlined above, the differentiable
penalty $s(\pmb\theta,\pmb t_s)$ is simply absorbed in the unregularized fitting
function $l(\pmb\theta)$. 

Now, for the non-differentiable part $p(\pmb\theta,\pmb p_s)$, the ista optimizer uses so-called proximal
operators. The details are beyond the scope here, but Parikh et al. (2013) provide
a very good overview of these algorithms. To make things work with ista, we
must pass such a proximal operator to the optimizer. Within **lesstimate**, we
have prepared a few of these proximal operators for well-known penalty functions.
Additionally, we need a function which returns the actual
penalty value. This is the penalty object in the function call. Finally, the 
penalty $p(\pmb\theta,\pmb t_p)$ gets its tuning parameters $\pmb t_p$. This
is the `tuningParameters` object above. To make things more concrete, let's look
at the elastic net again. In this case, penalty would be of class `less::penaltyLASSO` and the proximal operator of type `less::proximalOperatorLasso`.
The tuning parameters would again be $\alpha$ and $\lambda$ (and the
weights vector).

Note that many of the penalty function implemented in **lesstimate** are
typically not combined with a smooth penalty (e.g., scad, mcp, ...). In this
case, you must still pass a smoothPenalty object to ista. To this end,
we created the `less::noSmoothPenalty` class which can be used instead of
a smooth penalty like ridge. 

Finally, there is the control argument. This argument lets us fine tune the 
optimizer. To use the control argument, create a new control object as
follows:


```
less::control myControlObject = less::controlDefault();
```

Now, you can tweak each element of myControlObject; e.g.,


```
myControlObject.L0 = .9
```
