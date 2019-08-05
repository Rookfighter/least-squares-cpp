/* parabolic_error.cpp
 *
 *  Created on: 11 Dec 2018
 *      Author: Fabian Meyer
 */

#include <lsqcpp.h>

struct ParabolicError
{
    void operator()(const Eigen::VectorXd &xval,
        Eigen::VectorXd &fval,
        Eigen::MatrixXd &) const
    {
        fval.resize(xval.size() / 2);
        for(lsq::Index i = 0; i < fval.size(); ++i)
            fval(i) = xval(i*2) * xval(i*2) + xval(i*2+1) * xval(i*2+1);
    }
};

int main()
{
    lsq::GaussNewton<double, ParabolicError, lsq::WolfeBacktracking<double>> optimizer;

    optimizer.setMaxIterations(20);

    optimizer.setMinStepLength(1e-3);
    optimizer.setMinGradientLength(1e-3);
    optimizer.setMinError(0);
    optimizer.setVerbosity(4);

    Eigen::VectorXd initialGuess(2);
    initialGuess << 2, 2;

    auto result = optimizer.minimize(initialGuess);

    std::cout << "Done! Converged: " << (result.converged ? "true" : "false")
        << " Iterations: " << result.iterations << std::endl;

    // do something with final function value
    std::cout << "Final fval: " << result.fval.transpose() << std::endl;

    // do something with final x-value
    std::cout << "Final xval: " << result.xval.transpose() << std::endl;

    return 0;
}
