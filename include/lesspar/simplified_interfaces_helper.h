#ifndef SIMPLIFIED_HELPER_FUNCTIONS
#define SIMPLIFIED_HELPER_FUNCTIONS
// The optimizers implemented in lesspar are fairly flexible, resulting
// in a complexity overhead for users who just want to use one specific penalty
// function out of the box. The following classes are meant to reduce this
// overhead by providing simplified interfaces to specific penalty functions.

#include "common_headers.h"
#include "glmnet_class.h"
#include "glmnet_penalties.h"
#include "ista_class.h"
#include "ista_penalties.h"

namespace lessSEM
{

  /**
   * @brief We want to allow users to pass vectors of size 1 which are then resized to
   * the correct lenght. The following two functions achieve this for std::vector
   * and arma::rowvec.
   *
   * @tparam vecobj type of the std::vector
   * @param numberParameters target length of vector
   * @param userVector  user provided vector
   * @return std::vector<vecobj> vector of length numberParameters, where, if userVector was of lenght 1,
   * all elements are replaced with the single element provided by the user. Otherwise
   * the vector is returned without any changes.
   */
  template <typename vecobj>
  inline std::vector<vecobj> resizeVector(unsigned int numberParameters,
                                          std::vector<vecobj> userVector)
  {
    if (userVector.size() == 1)
    {
      vecobj userObj = userVector.at(0);
      userVector.resize(numberParameters);
      std::fill(userVector.begin(), userVector.end(), userObj);
    }
    return (userVector);
  }

  /**
   * @brief resize a vector
   *
   * @param numberParameters number of parameters of the target vector
   * @param userVector user supplied vector
   * @return arma::rowvec vector of length numberParameters
   */
  inline arma::rowvec resizeVector(unsigned int numberParameters,
                                   arma::rowvec userVector)
  {
    if (userVector.size() == 1)
    {
      double userObj = userVector(0);
      userVector.resize(numberParameters);
      userVector.fill(userObj);
    }
    return (userVector);
  }

  /**
   * @brief checks if all elements of a vector with unsigned integers are the same.
   *
   * @param myvec vector with unsigned integers
   * @return true if all are equal
   * @return false otherwise
   */
  inline bool allEqual(std::vector<unsigned int> myvec)
  {
    if (myvec.size() == 0)
    {
      error("Empty vector");
    }
    unsigned int element_1 = myvec.at(0);
    for (auto i : myvec)
    {
      if (i != element_1)
        return (false);
    }
    return (true);
  }

  /**
   * @brief Translates a vector with strings to the internal penalty type representation.
   *
   * @param penalty string vector with penalty name
   * @return std::vector<penaltyType> with penalty types
   */
  inline std::vector<penaltyType> stringPenaltyToPenaltyType(std::vector<std::string> penalty)
  {

    std::vector<penaltyType> penalties(penalty.size());

    for (unsigned int i = 0; i < penalty.size(); i++)
    {

      if (penalty.at(i).compare("none") == 0)
      {
        penalties.at(i) = penaltyType::none;
      }
      else if (penalty.at(i).compare("cappedL1") == 0)
      {
        penalties.at(i) = penaltyType::cappedL1;
      }
      else if (penalty.at(i).compare("lasso") == 0)
      {
        penalties.at(i) = penaltyType::lasso;
      }
      else if (penalty.at(i).compare("lsp") == 0)
      {
        penalties.at(i) = penaltyType::lsp;
      }
      else if (penalty.at(i).compare("mcp") == 0)
      {
        penalties.at(i) = penaltyType::mcp;
      }
      else if (penalty.at(i).compare("scad") == 0)
      {
        penalties.at(i) = penaltyType::scad;
      }
      else
      {
        error("Unknown penalty type: " +
              penalty.at(i) +
              ". Supported are: none, cappedL1, lasso, lsp, mcp, or scad.");
      }
    }

    return (penalties);
  }

  /**
   * @brief prints information about the penalties if verbose is set to true.
   *
   * @param parameterLabels stringVector with the names of the parameters
   * @param penalties penaltyType vector indicating the penalty for each parameter
   * @param lambda lambda tuning parameter values. One lambda value for each parameter
   * @param theta theta tuning parameter values. One theta value for each parameter
   * @return nothing
   */
  inline void printPenaltyDetails(
      const stringVector &parameterLabels,
      const std::vector<penaltyType> &penalties,
      const arma::rowvec &lambda,
      const arma::rowvec &theta)
  {

    std::vector<std::string> parameterLabels_(penalties.size());
    if ((unsigned int)parameterLabels.size() != (unsigned int)penalties.size())
    {
      for (unsigned int i = 0; i < penalties.size(); i++)
      {
        parameterLabels_.at(i) = std::to_string(i + 1);
      }
    }
    else
    {
      for (unsigned int i = 0; i < penalties.size(); i++)
      {
        parameterLabels_.at(i) = parameterLabels.at(i);
      }
    }

    for (unsigned int i = 0; i < penalties.size(); i++)
    {

      switch (penalties.at(i))
      {
      case none:
        print << "No penalty on "
              << parameterLabels_.at(i)
              << std::endl;
        break;
      case cappedL1:
        print << "cappedL1 penalty on "
              << parameterLabels_.at(i)
              << " lambda = " << lambda(i)
              << " theta = " << theta(i)
              << std::endl;
        break;
      case lasso:
        print << "lasso penalty on "
              << parameterLabels_.at(i)
              << " lambda = " << lambda(i)
              << std::endl;
        break;
      case lsp:
        print << "lsp penalty on "
              << parameterLabels_.at(i)
              << " lambda = " << lambda(i)
              << " theta = " << theta(i)
              << std::endl;
        break;
      case mcp:
        print << "mcp penalty on "
              << parameterLabels_.at(i)
              << " lambda = " << lambda(i)
              << " theta = " << theta(i)
              << std::endl;
        break;
      case scad:
        print << "scad penalty on "
              << parameterLabels_.at(i)
              << " lambda = " << lambda(i)
              << " theta = " << theta(i)
              << std::endl;
        break;
      default:
        error("Unknown penalty on " + parameterLabels_.at(i));
      }
    }
  }
}
#endif