#ifndef PENALTYTYPE_H
#define PENALTYTYPE_H

namespace lessSEM{
  /**
   * @brief specify the penalty type
   * 
   */
  enum penaltyType{
    none, ///> no penalty
    cappedL1, ///> cappedL1 penalty
    lasso, ///> lasso penalty
    lsp, /// lsp penalty
    mcp, ///> mcp penalty
    scad ///> scad penalty
  };
  const std::vector<std::string> penaltyType_txt = {
    "none",
    "cappedL1",
    "lasso",
    "lsp",
    "mcp",
    "scad"
  };
}
#endif
