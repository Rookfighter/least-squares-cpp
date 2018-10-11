# least-squares-cpp

![Cpp11](https://img.shields.io/badge/C%2B%2B-11-blue.svg)
![License](https://img.shields.io/packagist/l/doctrine/orm.svg)
![Travis Status](https://travis-ci.org/Rookfighter/least-squares-cpp.svg?branch=master)
![Appveyer Status](https://ci.appveyor.com/api/projects/status/y62egiabuk9ubie4?svg=true)

least-squares-cpp is a header-only C++ library for unconstrained non-linear least squares optimization. It provides newton type optimization algorithms such as:

* Gauss Newton
* Gradient Descent
* Levenberg Marquardt

The library also includes various line search algorithms:

* Armijo Backtracking
* ~Wolfe Linesearch~ TBD

## Install

Simply copy the header files into your project or install them using
the CMake build system by typing

```bash
cd path/to/repo
mkdir build
cd build
cmake ..
make install
```

The library requires Eigen3 to be installed on your system.
In Debian based systems you can simply type

```bash
apt-get install libeigen3-dev
```

Make sure Eigen3 can be found by your build system.

## Usage

There are three major steps to use least-squares-cpp:

* Implement your error function(s)
* Pick the optimization algorithm of your choice
* Pick the line search algorithm of your choice
