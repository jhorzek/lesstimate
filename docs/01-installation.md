# Installation

Being a header-only library, **lesstimate** is fairly easy to install.
However, depending on your use-case, the dependencies of **lesstimate** have 
to be installed differently.

## Using lesstimate in your R package

> We provide a [package template using **lesstimate**](https://github.com/jhorzek/lesstimateTemplateR). All procedures outlined in the following are already implemented in this template.

When using **lesstimate** in your R package, first make sure that **RcppArmadillo**
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

Clone the lesstimate git repository in your include folder with `git clone https://github.com/jhorzek/lesstimate.git`.

Add a file called `lessSEM.h` in the folder inst/include ([see here for an example](https://github.com/jhorzek/lesstimateTemplateR/blob/main/inst/include/lessSEM.h)) and
save the following in this file:

```
#ifndef lessSEM_H
#define lessSEM_H

#include "lesstimate/include/lesstimate.h"

#endif
```

This file makes sure that R can find the functions of **lesstimate**. 

In your src-folder, open the files Makevars and Makevars.win. Add the following line ([see here](https://github.com/jhorzek/lesstimateTemplateR/blob/main/src/Makevars)):
```
PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS) -I../inst/include/
```
This ensures that the include folder is found when compiling the package.

Next add `^lesstimateConfig-cmake$` to .Rbuildignore.

Open the DESCRIPTION file of your package and add `Rcpp` and `RcppArmadillo` to the field `LinkingTo` ([see here](https://github.com/jhorzek/lesstimateTemplateR/blob/main/DESCRIPTION)).

## Using lesstimate in your C++ package

> We provide a [template using **lesstimate** with C++](https://github.com/jhorzek/lesstimateTemplateCpp). All procedures outlined in the following are already implemented in this template. We 
highly recommend that you use this template as a starting point.

For C++, you will need the [**armadillo**](https://arma.sourceforge.net/) - library (Sanderson et al., 2016). **lesstimate** uses the [**cpm.Cmake**](https://github.com/cpm-cmake/CPM.cmake)
package manager to handle its dependencies. However, not all dependencies will be installed automatically. **armadillo** requires **BLAS** and **LAPACK**, both of which have to be installed on
the system. Instructions to install dependencies of **armadillo** are provided in the [documentation of the package](https://arma.sourceforge.net/download.html). 

# References

- Eddelbuettel D, Sanderson C (2014). “RcppArmadillo: Accelerating R with high-performance C++ linear algebra.” Computational Statistics and Data Analysis, 71, 1054–1063. doi:10.1016/j.csda.2013.02.005.
- Sanderson C, Curtin R (2016). Armadillo: a template-based C++ library for linear algebra. Journal of Open Source Software, 1 (2), pp. 26.

