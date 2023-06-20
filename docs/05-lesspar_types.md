# lesstimate types

## USE_R

The USE_R variable allows us to switch between the R implementation
and C++ without R. This variable can either be passed to the compiler
using -DUSE_R=1 or -DUSE_R=0 or by changing the value below. By default,
lesstimate will assume that you are using R and therefor set USE_R=1
if the variable is not defined otherwise.


## USE_R = 1

### stringVector

Identical to Rcpp::NumericVector

### stringVector

Identical to Rcpp::StringVector

## USE_R = 0

### stringVector

Provides similar functionality as Rcpp::StringVector.

#### Constructors:

- stringVector() {} 
- stringVector(std::vector<std::string> values_) : values(values_) {}

#### Fields

- **field** values: the values are a std::vector<std::string>

#### Methods

**at** 

returns the element at a specific location of the string

- **param** where: location
- **return** auto&

**size** and **length**

- **return** returns the number of elements in a stringVector

**fill**

Fill all elements of the stringVector with the same string

- **param** with: string to fill elements with

### toStringVector

cast std::vector<std::string> to stringVector

- **param** vec: std::vector<std::string>
- **return** stringVector

### numericVector

numericVector provides functionality similar to Rcpp::NumericVector

#### Constructor

- numericVector()
- numericVector(int n_elem)
- numericVector(arma::rowvec vec)
- numericVector(arma::rowvec vec, std::vector<std::string> labels)

#### Fields

- **field** values: arma::rowvec
- **field** labels: labels for parameters

#### Methods

**at** 

returns the element at a specific location of the string

- **param** where: location
- **return** auto&

**()**

- identical to **at**

**size** and **length**

- **return** returns the number of elements in a stringVector

**fill**

Fill all elements of the stringVector with the same string

- **param** with: string to fill elements with

**names**

access the names of the numericVector elements

- **return** stringVector&

### toArmaVector

Cast numericVector to arma::rowvec

- **param** numVec: numericVector
- **return** arma::rowvec

### toNumericVector

Cast arma::rowvec to numericVector

- **param** vec: arma::rowvec
- **return** numericVector
