#pragma once

// The USE_R variable allows us to switch between the R implementation
// and C++ without R. This variable can either be passed to the compiler
// using -DUSE_R=1 or -DUSE_R=0 or by changing the value below. By default,
// lessOptimizers will assume that you are using R and therefor set USE_R=1
// if the variable is not defined otherwise.
#ifndef USE_R
#define USE_R 1
#endif

#if USE_R
// ----------------------
// USING R
// Here, we rely on Rcpp and RcppArmadillo
// ----------------------

// include necessary headers:
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// define print, warnings, and errors:
#define print Rcpp::Rcout
#define warn Rcpp::warning
#define error Rcpp::stop

namespace lessSEM
{
  // define data types:
  typedef Rcpp::NumericVector numericVector;
  typedef Rcpp::StringVector stringVector;

  // define functions:
  inline arma::rowvec toArmaVector(numericVector numVec)
  {
    return (Rcpp::as<arma::rowvec>(numVec));
  }

  inline numericVector sample(numericVector vec, int nSamples, bool replace)
  {
    return (Rcpp::sample(vec, nSamples, replace));
  }

  inline numericVector unif(int n, double min, double max)
  {
    return (Rcpp::runif(n, min, max));
  }
}

#else

// ----------------------
// NOT USING R
// Here, we replace all Rcpp-specific elements with
// armadillo and custom classes that imitate those of
// Rcpp.
// ----------------------

// include headers:
#include <armadillo>

// define print, warnings, and errors:
#define print std::cout
namespace lessSEM
{
  inline void warn(std::string warningMessage)
  {
    std::cout << "WARNING: " + warningMessage;
  }
}
#define error throw std::invalid_argument

// NA
#define NA_REAL arma::datum::nan

// define data types:
namespace lessSEM
{
  class stringVector
  {
  public:
    std::vector<std::string> values;

    stringVector() {}
    stringVector(int n_elem)
    {
      values.resize(n_elem);
    }

    stringVector(std::vector<std::string> values_) : values(values_) {}

    auto &at(unsigned int where)
    {
      return (values.at(where));
    }
    const auto &at(unsigned int where) const
    {
      return (values.at(where));
    }

    int size()
    {
      return (values.size());
    }

    const int size() const
    {
      return (values.size());
    }

    int length()
    {
      return (values.size());
    }

    void fill(std::string with)
    {
      for (auto &value : values)
        value = with;
    }
  };

  class numericVector
  {
  public:
    arma::rowvec values;
    stringVector par_names;

    numericVector()
    {
      values.resize(0);
      par_names.values.resize(0);
    }

    numericVector(int n_elem)
    {
      values.resize(n_elem);
      par_names.values.resize(n_elem);
      for (int i = 0; i < n_elem; i++)
        par_names.values.at(i) = "";
    }

    numericVector(arma::rowvec vec)
    {
      values = vec;
      par_names.values.resize(values.n_elem);
      for (int i = 0; i < values.n_elem; i++)
        par_names.values.at(i) = "";
    }

    numericVector(arma::rowvec vec, std::vector<std::string> labels)
    {
      values = vec;
      par_names.values = labels;
    }

    // methods
    double &at(unsigned int position)
    {
      return (values(position));
    }

    double &operator()(unsigned int position)
    {
      return (this->at(position));
    }

    friend std::ostream &operator<<(std::ostream &output, const numericVector &numVec)
    {
      output << numVec.values;
      return (output);
    }

    stringVector &names()
    {
      return par_names;
    }

    int length()
    {
      return (values.n_elem);
    }

    void fill(const double value)
    {
      values.fill(value);
    }
  };

  // define functions:

  inline arma::rowvec toArmaVector(numericVector numVec)
  {
    return (numVec.values);
  }

  inline numericVector sample(numericVector vec, int nSamples, bool replace)
  {
    arma::rowvec values = vec.values;
    std::vector<std::string> labels = vec.par_names.values;
    arma::rowvec positions(vec.length());
    for (int i = 0; i < vec.length(); i++)
      positions(i) = i;

    positions = arma::shuffle(positions);
    for (unsigned int i = 0; i < positions.n_elem; i++)
    {
      vec.values(i) = values(positions(i));
      vec.par_names.values.at(i) = labels.at(i);
    }
    return (vec);
  }

  inline numericVector unif(int n, double min, double max)
  {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> unif_dist(min, max);
    arma::vec tst(n);
    numericVector ret(n);
    for (int i = 0; i < n; i++)
      ret(i) = unif_dist(generator);
    return (ret);
  }
}

#endif
