# Optimization Cpp

Optimization Cpp is a basic C++ library for Nonlinear Optimization. It provides Newton type optimization algorithms such as:

* Gauss Newton
* Gradient Descent
* Levenberg Marquardt

The library also includes various line search algorithms:

* Armijo Backtracking
* ~Wolfe Linesearch~ TBD

## Install

First download the dependencies locally as git submodules.

```bash
cd <path-to-repo>
git submodule update --init --recursive
```

Then the library can be built using the CMake build system by running

```bash
cd <path-to-repo>
mkdir build
cd build
cmake ..
make
```

You can also copy the repo into your project (possibly as git submodule) and include it into your CMake project with the command

```CMake
add_subdirectory("optimization-cpp")
```

The variables ```OPTCPP_INCLUDE_DIR``` and ```OPTCPP_LIBRARY``` should then be available in your project.

Or you can simply copy the source into your project and build it with the build system of your choice.

## Usage
