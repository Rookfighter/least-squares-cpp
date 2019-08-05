# least-squares-cpp

![Cpp11](https://img.shields.io/badge/C%2B%2B-11-blue.svg)
![License](https://img.shields.io/packagist/l/doctrine/orm.svg)
![Travis Status](https://travis-ci.org/Rookfighter/least-squares-cpp.svg?branch=master)
![Appveyer Status](https://ci.appveyor.com/api/projects/status/y62egiabuk9ubie4?svg=true)

```least-squares-cpp``` is a header-only C++ library for unconstrained non-linear
least squares optimization using the ```Eigen3``` library.

It provides the following newton type optimization algorithms:

* Gradient Descent
* Gauss Newton
* Levenberg Marquardt

The library also includes the following step size calculation methods:

* Constant Step Size
* Barzilai-Borwein Method
* Armijo Backtracking
* Wolfe Backtracking

## Install

Simply copy the header file into your project or install it using
the CMake build system by typing

```bash
cd path/to/repo
mkdir build
cd build
cmake ..
make install
```

The library requires ```Eigen3``` to be installed on your system.
In Debian based systems you can simply type

```bash
apt-get install libeigen3-dev
```

Make sure ```Eigen3``` can be found by your build system.

You can use the CMake Find module in ```cmake/``` to find the installed header.

## Usage

There are three major steps to use ```least-squares-cpp```:

* Implement your error function as functor
* Instantiate the optimization algorithm of your choice
* Pick the line search algorithm and parameters of your choice

```cpp
#include <lsqcpp.h>

// Implement an objective functor.
struct ParabolicError
{
    void operator()(const Eigen::VectorXd &xval,
        Eigen::VectorXd &fval,
        Eigen::MatrixXd &) const
    {
        // omit calculation of jacobian, so finite differences will be used
        // to estimate jacobian numerically
        fval.resize(xval.size() / 2);
        for(lsq::Index i = 0; i < fval.size(); ++i)
            fval(i) = xval(i*2) * xval(i*2) + xval(i*2+1) * xval(i*2+1);
    }
};

int main()
{
    // Create GaussNewton optimizer object with ParabolicError functor as objective.
    // There are GradientDescent, GaussNewton and LevenbergMarquardt available.
    //
    // You can specify a StepSize functor as template parameter.
    // There are ConstantStepSize, BarzilaiBorwein, ArmijoBacktracking
    // WolfeBacktracking available. (Default for GaussNewton is ArmijoBacktracking)
    //
    // You can additionally specify a Callback functor as template parameter.
    //
    // You can additionally specify a FiniteDifferences functor as template
    // parameter. There are Forward-, Backward- and CentralDifferences
    // available. (Default is CentralDifferences)
    //
    // For GaussNewton and LevenbergMarquardt you can additionally specify a
    // linear equation system solver.
    // There are DenseSVDSolver and DenseCholeskySolver available.
    lsq::GaussNewton<double, ParabolicError, lsq::ArmijoBacktracking<double>> optimizer;

    // Set number of iterations as stop criterion.
    // Set it to 0 or negative for infinite iterations (default is 0).
    optimizer.setMaxIterations(100);

    // Set the minimum length of the gradient.
    // The optimizer stops minimizing if the gradient length falls below this
    // value.
    // Set it to 0 or negative to disable this stop criterion (default is 1e-9).
    optimizer.setMinGradientLength(1e-6);

    // Set the minimum length of the step.
    // The optimizer stops minimizing if the step length falls below this
    // value.
    // Set it to 0 or negative to disable this stop criterion (default is 1e-9).
    optimizer.setMinStepLength(1e-6);

    // Set the minimum least squares error.
    // The optimizer stops minimizing if the error falls below this
    // value.
    // Set it to 0 or negative to disable this stop criterion (default is 0).
    optimizer.setMinError(0);

    // Set the the parametrized StepSize functor used for the step calculation.
    optimizer.setStepSize(lsq::ArmijoBacktracking<double>(0.8, 0.1, 1e-10, 1.0, 0));

    // Turn verbosity on, so the optimizer prints status updates after each
    // iteration.
    optimizer.setVerbosity(2);

    // Set initial guess.
    Eigen::VectorXd initialGuess(4);
    initialGuess << 1, 2, 3, 4;

    // Start the optimization.
    auto result = optimizer.minimize(initialGuess);

    std::cout << "Done! Converged: " << (result.converged ? "true" : "false")
        << " Iterations: " << result.iterations << std::endl;

    // do something with final function value
    std::cout << "Final fval: " << result.fval.transpose() << std::endl;

    // do something with final x-value
    std::cout << "Final xval: " << result.xval.transpose() << std::endl;

    return 0;
}
```
