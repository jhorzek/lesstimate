#pragma once

/**
 * @brief The USE_R variable allows us to switch between the R implementation
 * and C++ without R. This variable can either be passed to the compiler
 * using -DUSE_R=1 or -DUSE_R=0 or by changing the value below. By default,
 * lesstimate will assume that you are using R and therefor set USE_R=1
 * if the variable is not defined otherwise.
 */
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

  /**
   * @brief cast a Rcpp::NumericVector to arma::rowvec
   *
   * @param numVec vector of class Rcpp::NumericVector
   * @return arma::rowvec
   */
  inline arma::rowvec toArmaVector(numericVector numVec)
  {
    arma::rowvec armaVec(numVec.length());
    for (unsigned int i = 0; i < numVec.length(); i++)
    {
      armaVec(i) = numVec(i);
    }
    return (armaVec);
  }

  /**
   * @brief cast an arma::rowvec to Rcpp::NumericVector
   *
   * @param vec vector of class arma::rowvec
   * @return numericVector
   */
  inline numericVector toNumericVector(arma::rowvec vec)
  {
    numericVector numVec(vec.n_elem);
    for (unsigned int i = 0; i < numVec.length(); i++)
    {
      numVec(i) = vec(i);
    }
    return (numVec);
  }

  /**
   * @brief cast a std::string - vector to Rcpp::StringVector
   *
   * @param vec vector of class std::vector<std::string>
   * @return stringVector
   */
  inline stringVector toStringVector(std::vector<std::string> vec)
  {
    stringVector myStringVec(vec.size());
    for (unsigned int i = 0; i < myStringVec.length(); i++)
    {
      myStringVec(i) = vec.at(i);
    }
    return (myStringVec);
  }

  /**
   * @brief sample randomly elements from a vector
   *
   * @param vec Rcpp::NumericVector with values to sample from
   * @param nSamples number of elements to draw randomly
   * @param replace should elements be replaced (i.e., can elements be drawn multiple times)
   * @return numericVector
   */
  inline numericVector sample(numericVector vec, int nSamples, bool replace)
  {
    return (Rcpp::sample(vec, nSamples, replace));
  }

  /**
   * @brief sample random values from a uniform distribution
   *
   * @param n number of elements to draw
   * @param min minimum value of the uniform distribution
   * @param max maximum value of the uniform distribution
   * @return numericVector
   */
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

class CustomException: public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

template <typename... Args>
[[noreturn]] void error(Args&&... args) {
    throw CustomException(std::forward<Args>(args)...);
}

// NA
#define NA_REAL arma::datum::nan

// define data types:
namespace lessSEM
{
  /**
   * @brief Provides similar functionality as Rcpp::StringVector.
   *
   */
  class stringVector
  {
  public:
    /**
     * @brief the values are a std::vector<std::string>
     *
     */
    std::vector<std::string> values;

    /**
     * @brief Construct a new string Vector object
     *
     */
    stringVector() {}

    /**
     * @brief Construct a new string Vector object
     *
     * @param n_elem length of the vector
     */
    stringVector(int n_elem)
    {
      values.resize(n_elem);
    }

    /**
     * @brief Construct a new string Vector object
     *
     * @param values_ provide the strings to be stored in the stringVector
     */
    stringVector(std::vector<std::string> values_) : values(values_) {}

    /**
     * @brief return the element at a specific location of the string
     *
     * @param where location
     * @return auto&
     */
    auto &at(unsigned int where)
    {
      return (values.at(where));
    }

    /**
     * @brief return the element at a specific location of the string
     *
     * @param where location
     * @return const auto&
     */
    const auto &at(unsigned int where) const
    {
      return (values.at(where));
    }

    /**
     * @brief returns the number of elements in a stringVector
     *
     * @return int
     */
    int size()
    {
      return (values.size());
    }

    /**
     * @brief returns the number of elements in a stringVector
     *
     * @return const int
     */
    const int size() const
    {
      return (values.size());
    }

    /**
     * @brief returns the number of elements in a stringVector
     *
     * @return int
     */
    int length()
    {
      return (values.size());
    }

    /**
     * @brief fill all elements of the stringVector with the same string
     *
     * @param with string to fill elements with
     */
    void fill(std::string with)
    {
      for (auto &value : values)
        value = with;
    }
  };

  /**
   * @brief cast std::vector<std::string> to stringVector
   *
   * @param vec std::vector<std::string>
   * @return stringVector
   */
  inline stringVector toStringVector(std::vector<std::string> vec)
  {
    stringVector myStringVec;
    myStringVec.values = vec;
    return (myStringVec);
  }

  /**
   * @brief numericVector provides functionality similar to Rcpp::NumericVector
   *
   */
  class numericVector
  {
  public:
    /**
     * @brief values are stored as arma::rowvec
     *
     */
    arma::rowvec values;
    /**
     * @brief labels are stored as stringVector
     *
     */
    stringVector par_names;

    /**
     * @brief Construct a new numeric Vector object
     *
     */
    numericVector()
    {
      values.resize(0);
      par_names.values.resize(0);
    }

    /**
     * @brief Construct a new numeric Vector object
     *
     * @param n_elem length of vector
     */
    numericVector(int n_elem)
    {
      values.resize(n_elem);
      par_names.values.resize(n_elem);
      for (int i = 0; i < n_elem; i++)
        par_names.values.at(i) = "";
    }

    /**
     * @brief Construct a new numeric Vector object
     *
     * @param vec arma::rowvec with values to be stored in numericVector
     */
    numericVector(arma::rowvec vec)
    {
      values = vec;
      par_names.values.resize(values.n_elem);
      for (unsigned long long int i = 0; i < values.n_elem; i++)
        par_names.values.at(i) = "";
    }

    /**
     * @brief Construct a new numeric Vector object
     *
     * @param vec arma::rowvec with values to be stored in numericVector
     * @param labels std::vector<std::string> with names of the elements in numericVector
     */
    numericVector(arma::rowvec vec, std::vector<std::string> labels)
    {
      values = vec;
      par_names.values = labels;
    }

    /**
     * @brief return the element at a specific position in the vector
     *
     * @param position position of element
     * @return double&
     */
    double &at(unsigned int position)
    {
      return (values(position));
    }

    /**
     * @brief return the element at a specific position in the vector
     *
     * @param position position of element
     * @return double&
     */
    double &operator()(unsigned int position)
    {
      return (this->at(position));
    }

    /**
     * @brief print elements of numericVector
     *
     * @param output std::ostream
     * @param numVec numericVector
     * @return std::ostream&
     */
    friend std::ostream &operator<<(std::ostream &output, const numericVector &numVec)
    {
      output << numVec.values;
      return (output);
    }

    /**
     * @brief access the names of the numericVector elements
     *
     * @return stringVector&
     */
    stringVector &names()
    {
      return par_names;
    }

    /**
     * @brief get number of elements in the numeric vector
     *
     * @return int
     */
    int length()
    {
      return (values.n_elem);
    }

    /**
     * @brief fill all elements in the numeric vector with the same value
     *
     * @param value value to fill the vector with
     */
    void fill(const double value)
    {
      values.fill(value);
    }
  };

  // define functions:
  /**
   * @brief cast numericVector to arma::rowvec
   *
   * @param numVec numericVector
   * @return arma::rowvec
   */
  inline arma::rowvec toArmaVector(numericVector numVec)
  {
    return (numVec.values);
  }

  /**
   * @brief cast arma::rowvec to numericVector
   *
   * @param vec arma::rowvec
   * @return numericVector
   */
  inline numericVector toNumericVector(arma::rowvec vec)
  {
    return (numericVector(vec));
  }

  /**
   * @brief sample randomly elements from a vector
   *
   * @param vec Rcpp::NumericVector with values to sample from
   * @param nSamples number of elements to draw randomly
   * @param replace should elements be replaced (i.e., can elements be drawn multiple times)
   * @return numericVector
   */
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

  /**
   * @brief sample random values from a uniform distribution
   *
   * @param n number of elements to draw
   * @param min minimum value of the uniform distribution
   * @param max maximum value of the uniform distribution
   * @return numericVector
   */
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
