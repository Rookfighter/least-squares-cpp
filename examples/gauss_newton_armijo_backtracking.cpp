/// gauss_newton_armijo_backtracking.cpp
///
/// Created on: 03 Aug 2019
/// Author:     Fabian Meyer
/// License:    MIT

#include <lsqcpp/lsqcpp.hpp>

// Implement an objective functor.
struct ParabolicError
{
    static constexpr bool ComputesJacobian = false;

    template<typename Scalar, int Inputs, int Outputs>
    void operator()(const Eigen::Matrix<Scalar, Inputs, 1> &xval,
                    Eigen::Matrix<Scalar, Outputs, 1> &fval) const
    {
        // omit calculation of jacobian, so finite differences will be used
        // to estimate jacobian numerically
        fval.resize(xval.size() / 2);
        for(lsqcpp::Index i = 0; i < fval.size(); ++i)
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
    lsqcpp::GaussNewtonX<double, ParabolicError, lsqcpp::ArmijoBacktracking> optimizer;

    // Set number of iterations as stop criterion.
    // Set it to 0 or negative for infinite iterations (default is 0).
    optimizer.setMaximumIterations(100);

    // Set the minimum length of the gradient.
    // The optimizer stops minimizing if the gradient length falls below this
    // value.
    // Set it to 0 or negative to disable this stop criterion (default is 1e-9).
    optimizer.setMinimumGradientLength(1e-6);

    // Set the minimum length of the step.
    // The optimizer stops minimizing if the step length falls below this
    // value.
    // Set it to 0 or negative to disable this stop criterion (default is 1e-9).
    optimizer.setMinimumStepLength(1e-6);

    // Set the minimum least squares error.
    // The optimizer stops minimizing if the error falls below this
    // value.
    // Set it to 0 or negative to disable this stop criterion (default is 0).
    optimizer.setMinimumError(0);

    // Set the parameters of the step refiner (Armijo Backtracking).
    optimizer.setRefinementParameters({0.8, 1e-4, 1e-10, 1.0, 0});

    // Turn verbosity on, so the optimizer prints status updates after each
    // iteration.
    optimizer.setVerbosity(4);

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
