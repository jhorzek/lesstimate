# Installation

Being a header-only library, **lessOptimizers** is fairly easy to install.
However, depending on your use-case, the dependencies of **lessOptimizers** have 
to be installed differently.

## Using lessOptimizers in your R package

When using **lessOptimizers** in your R package, first make sure that **RcppArmadillo**
is installed (if not, run `install.packages("RcppArmadillo")`. You will also
need a C++ compiler. Instructions to install those can be found in the official
R documentation [ADD LINK HERE].

Next, create a folder called `inst` in your R package. Within `inst`, create a folder
called `include`. You should now have the following folder structure:


- R
- inst
	- include
- ...

Clone the lessOptimizers git repository in your include folder.

ADD CLONE COMMAND

ADD PATH TO lessOPTIMIZER TO PROJECT

ADD CMakeLists.txt to buildignore

ADD RcppArmadillo to LinkingTo in Description

SEE lessSEM for example project; ADD LINKS TO SPECIFIC LINES

## Using lessOptimizers in your C++ package

For C++, you will need the **armadillo** - library. This library is available from [ADD LINK TO ARMADILLO]
or using the package manager Conan or vcpkg. You will have to include lessOptimizers in your
C++ library Cmake file. An example can be found in the lessLMcpp-projec [ADD LINK].

