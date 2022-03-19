/// provide_explicit_jacobian.cpp
///
/// Created on: 11 Nov 2020
/// Author:     Fabian Meyer
/// License:    MIT

#include <lsqcpp/lsqcpp.hpp>

// Implement an objective functor.
struct ParabolicError
{
    constexpr static bool ComputesJacobian = true;

    template<typename Scalar, int Inputs, int Outputs>
    void operator()(const Eigen::Matrix<Scalar, Inputs, 1> &xval,
                    Eigen::Matrix<Scalar, Outputs, 1> &fval,
                    Eigen::Matrix<Scalar, Outputs, Inputs> &jacobian) const
    {
        assert(xval.size() % 2 == 0);

        // calculate the error vector
        fval.resize(xval.size() / 2);
        for(lsqcpp::Index i = 0; i < fval.size(); ++i)
            fval(i) = xval(i*2) * xval(i*2) + xval(i*2+1) * xval(i*2+1);

        // calculate the jacobian explicitly
        jacobian.setZero(fval.size(), xval.size());
        for(lsqcpp::Index i = 0; i < jacobian.rows(); ++i)
        {
            jacobian(i, i*2) = 2* xval(i*2);
            jacobian(i, i*2+1) = 2* xval(i*2+1);
        }
    }
};

int main()
{
    // Create GradienDescent optimizer with Barzilai Borwein method
    lsqcpp::GaussNewtonX<double, ParabolicError, lsqcpp::ArmijoBacktracking> optimizer;

    // Set number of iterations as stop criterion.
    optimizer.setMaximumIterations(100);

    // Set the minimum length of the gradient.
    optimizer.setMinimumGradientLength(1e-6);

    // Set the minimum length of the step.
    optimizer.setMinimumStepLength(1e-6);

    // Set the minimum least squares error.
    optimizer.setMinimumError(0);

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
