# Model

## model class

`model` is the base class used in every optimizer implemented in lesspar.
The user specified model should inherit from the model class and must implement
the two methods defined therein

### methods

#### fit

`fit` takes arguments parameterValues (arma::rowvec) and parameterLabels (stringVector; see common_headers.h)
specifying the parameter values and the labels of the paramters. The function should return the fit value (double).

- **param** parameterValues: numericVector with parameter values
- **param** parameterLabels: stringVector with parameterLabels
- **return** double

#### gradients

`gradients` takes arguments parameterValues(arma::rowvec) and parameterLabels(stringVector; see common_headers.h) specifying the parameter values and the labels of the paramters.The function should return the gradients(arma::rowvec)

- **param** parameterValues: numericVector with parameter values
- **param** parameterLabels: stringVector with parameterLabels
- **return** arma::rowvec gradients
