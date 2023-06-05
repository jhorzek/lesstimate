# Installation

Being a header-only library, **lessOptimizers** is fairly easy to install.
However, depending on your use-case, the dependencies of **lessOptimizers** have 
to be installed differently.

## Using lessOptimizers in your R package

When using **lessOptimizers** in your R package, first make sure that **RcppArmadillo**
(Eddelbuettel et al., 2014)
is installed (if not, run `install.packages("RcppArmadillo")`. You will also
need a C++ compiler. Instructions to [install C++](https://adv-r.hadley.nz/rcpp.html#prerequisites-17) 
can be found in the *Advanced R* book by Hadley Wickham.

Next, create a folder called `inst` in your R package. Within `inst`, create a folder
called `include`. You should now have the following folder structure:

```
|- R
|- inst
|    |- include
|- src
|    |- Makevars
|    |- Makevars.win
|- DESCRIPTION
|- .Rbuildignore
```

Clone the lessOptimizers git repository in your include folder with `git clone https://github.com/jhorzek/lessOptimizers.git`.

Add a file called `lessSEM.h` in the folder inst/include ([see here for an example](https://github.com/jhorzek/lessSEM/blob/82a4432649f4c9d6072f79836ef3ddefb001d083/inst/include/lessSEM.h)) and
save the following in this file:

```
#ifndef lessSEM_H
#define lessSEM_H

#include "lessOptimizers/include/lessOptimizers.h"

#endif
```

This file makes sure that R can find the functions of **lessOptimizers**. 

In your src-folder, open the files Makevars and Makevars.win. Add the following line ([see here](https://github.com/jhorzek/lessSEM/blob/82a4432649f4c9d6072f79836ef3ddefb001d083/src/Makevars#L1)):
```
PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS) -I../inst/include/
```
This ensures that the include folder is found when compiling the package.

Next add `^CMakeLists.txt$` to .Rbuildignore.

Open the DESCRIPTION file of your package and add `Rcpp` and `RcppArmadillo` to the field `LinkingTo` ([see here](https://github.com/jhorzek/lessSEM/blob/82a4432649f4c9d6072f79836ef3ddefb001d083/DESCRIPTION#L45)).

## Using lessOptimizers in your C++ package

For C++, you will need the **armadillo**(https://arma.sourceforge.net/) - library (Sanderson et al., 2016). This library is also available
using the package manager Conan or vcpkg. You will have to include lessOptimizers in your
C++ library Cmake file. An example can be found in the [lessLMcpp-project](https://github.com/jhorzek/lessLMcpp).

# References

- Eddelbuettel D, Sanderson C (2014). “RcppArmadillo: Accelerating R with high-performance C++ linear algebra.” Computational Statistics and Data Analysis, 71, 1054–1063. doi:10.1016/j.csda.2013.02.005.
- Sanderson C, Curtin R (2016). Armadillo: a template-based C++ library for linear algebra. Journal of Open Source Software, 1 (2), pp. 26.
