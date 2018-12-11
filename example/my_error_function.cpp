/*
 * my_error_function.cpp
 *
 *  Created on: 11 Dec 2018
 *      Author: Fabian Meyer
 */

#include <lsq/lsqcpp.h>
#include <iostream>

// implement your error function as sub class of lsq::ErrorFunction
class MyErrorFunction : public lsq::ErrorFunction<double>
{
public:
    lsq::Vectord values;

    void _evaluate(
        const lsq::Vectord &state,
        lsq::Vectord& outVal,
        lsq::Matrixd&) override
    {
        // implement your error function
        // if you do not calculate outJac (leave it untouched), finite
        // differences will be used to estimate it

        lsq::Vectord tmp = state;
        for(Eigen::Index i = 0; i < state.rows(); ++i)
            tmp[i] *= state[i];
        outVal = values - tmp;
    }
};

int main()
{
    // choose optimization algorithm e.g. gauss newton, levenberg marquardt, etc.
    lsq::GaussNewton<double> optalgo;

    // choose a line search algorithm
    // set to nullptr (default) to use none
    optalgo.setLineSearchAlgorithm(new lsq::ArmijoBacktracking<double>());

    // set stop conditions
    // set maximum number of iterations
    optalgo.setMaxIterations(20);
    // set epsilon for newton step length
    optalgo.setEpsilon(1e-6);

    // set verbosity
    optalgo.setVerbose(true);

    // set the error function
    MyErrorFunction *errFunc = new MyErrorFunction();
    errFunc->values.resize(4);
    errFunc->values << 1, 4, 9, 16;
    optalgo.setErrorFunction(errFunc);

    // choose initial state
    lsq::Vectord state = errFunc->values * 0.33;
    // optimize
    std::cout << "Optimize" << std::endl;
    auto result = optalgo.optimize(state);
    if(result.converged)
    {
        std::cout << "Converged!" << std::endl;
        std::cout << "-- iterations=" << result.iterations << std::endl;
        // do something with the resulting state
        std::cout << "-- state=[" << result.state.transpose() << ']' << std::endl;
        // or use the error value
        std::cout << "-- error=" << result.error << std::endl;
    }

    return 0;
}
