/* gradient_descent_barzilai_borwein.cpp
 *
 *  Created on: 11 Nov 2020
 *      Author: Fabian Meyer
 */

#include <lsqcpp.h>

// Implement an objective functor.
struct ParabolicError
{
    void operator()(const Eigen::VectorXd &xval,
        Eigen::VectorXd &fval,
        Eigen::MatrixXd &jacobian) const
    {
        // calculate the error vector
        fval.resize(xval.size() / 2);
        for(lsq::Index i = 0; i < fval.size(); ++i)
            fval(i) = xval(i*2) * xval(i*2) + xval(i*2+1) * xval(i*2+1);

        // calculate the jacobian explicitly
        jacobian.setZero(fval.size(), xval.size());
        for(lsq::Index i = 0; i < jacobian.rows(); ++i)
        {
            jacobian(i, i*2) = 2* xval(i*2);
            jacobian(i, i*2+1) = 2* xval(i*2+1);
        }
    }
};

int main()
{
    // Create GradienDescent optimizer with Barzilai Borwein method
    lsq::LevenbergMarquardt<double, ParabolicError> optimizer;

    // Set number of iterations as stop criterion.
    optimizer.setMaxIterations(100);

    // Set the minimum length of the gradient.
    optimizer.setMinGradientLength(1e-6);

    // Set the minimum length of the step.
    optimizer.setMinStepLength(1e-6);

    // Set the minimum least squares error.
    optimizer.setMinError(0);

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
