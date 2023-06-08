#ifndef PENALTYTYPE_H
#define PENALTYTYPE_H

namespace lessSEM{
  
  enum penaltyType{
    none,
    cappedL1,
    lasso,
    lsp,
    mcp,
    scad
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
