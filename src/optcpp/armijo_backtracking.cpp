/*
 * armijo_backtracking.cpp
 *
 *  Created on: 23 May 2018
 *      Author: Fabian Meyer
 */

#include "optcpp/armijo_backtracking.h"
#include "optcpp/linear_equation_system.h"

namespace opt
{
    static bool armijoCondition(const LinearEquationSystem& currLES,
        const LinearEquationSystem& refLES,
        const Eigen::VectorXd  &step,
        const double stepLen,
        const double gamma)
    {
        double currErr = currLES.b.norm();
        double linErr = (refLES.b + gamma * stepLen * refLES.A.transpose() *
            step).norm();

        return currErr < linErr;
    }

    ArmijoBacktracking::ArmijoBacktracking()
        : LineSearchAlgorithm(), beta_(0.8), gamma_(0.1), maxStepLen_(1.0)
    {

    }

    ArmijoBacktracking::~ArmijoBacktracking()
    {

    }

    void ArmijoBacktracking::setBeta(const double beta)
    {
        assert(beta_ > 0.0 && beta_ < 1.0);
        beta_ = beta;
    }

    void ArmijoBacktracking::setGamma(const double gamma)
    {
        assert(gamma > 0.0 && gamma < 0.5);
        gamma_ = gamma;
    }

    void ArmijoBacktracking::setMaxStepLen(const double stepLen)
    {
        maxStepLen_ = stepLen;
    }

    void ArmijoBacktracking::setMaxIterations(const size_t maxIt)
    {
        maxIt_ = maxIt;
    }

    double ArmijoBacktracking::stepLength(
        const Eigen::VectorXd &state,
        const Eigen::VectorXd &step,
        const std::vector<ErrorFunction *> &errFuncs) const
    {
        double result = maxStepLen_;
        LinearEquationSystem currLES(state + result * step, errFuncs);
        LinearEquationSystem refLES(state, errFuncs);

        size_t iterations = 0;
        // check for armijo condition
        while(!armijoCondition(currLES,refLES,step,result,gamma_) &&
            (maxIt_ == 0 || iterations < maxIt_))
        {
            // decrease step length
            result *= beta_;
            currLES.construct(state + result * step, errFuncs);
            ++iterations;
        }

        return result;
    }
}
