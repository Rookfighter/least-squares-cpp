/// gradient_descent_barzilai_borwein.cpp
///
///  Created on: 11 Nov 2020
///  Author:     Fabian Meyer
/// License:    MIT

#include <lsqcpp/lsqcpp.h>

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
    // Create GradienDescent optimizer with Barzilai Borwein method
    lsqcpp::GradientDescentX<double, ParabolicError, lsqcpp::BarzilaiBorwein> optimizer;

    // Set number of iterations as stop criterion.
    optimizer.setMaximumIterations(100);

    // Set the minimum length of the gradient.
    optimizer.setMinimumGradientLength(1e-6);

    // Set the minimum length of the step.
    optimizer.setMinimumStepLength(1e-6);

    // Set the minimum least squares error.
    optimizer.setMinimumError(0);

    // Set the parameters of the step refiner (BArzilai Borwein).
    optimizer.setStepRefiner({lsqcpp::BarzilaiBorwein::Mode::Inverse, 1e-2});

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
