# Optimization Cpp

Optimization Cpp is a basic C++ library for unconstrained nonlinear least squares optimization. It provides Newton type optimization algorithms such as:

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

Then build the library with CMake by running

```bash
cd <path-to-repo>
mkdir build
cd build
cmake ..
make
```

Or you can simply copy the source into your project and build it with the build system of your choice.

## Usage
