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
    static bool armijoCondition(const double currVal,
                                const double refVal,
                                const Eigen::VectorXd  &refGrad,
                                const Eigen::VectorXd  &step,
                                const double stepLen,
                                const double gamma)
    {
        assert(refGrad.size() == step.size());
        return currVal <= refVal + gamma * stepLen * (refGrad.transpose() *
                         step)(0);
    }

    ArmijoBacktracking::ArmijoBacktracking()
        : LineSearchAlgorithm(), beta_(0.8), gamma_(0.1), minStepLen_(1e-4),
        maxStepLen_(1.0), maxIt_(0)
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

    void ArmijoBacktracking::setBounds(const double minLen, const double maxLen)
    {
        assert(minLen < maxLen);
        maxStepLen_ = maxLen;
        minStepLen_ = minLen;
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
        double refVal = refLES.b.norm();
        double currVal = currLES.b.norm();
        Eigen::VectorXd refGrad = refLES.A.transpose() * refLES.b;

        // ensure step is descent direction
        assert(refGrad.size() == step.size());
        assert((refGrad.transpose() * step)(0) < 0);

        size_t iterations = 0;
        // check for armijo condition
        while(!armijoCondition(currVal, refVal, refGrad, step, result, gamma_)
            && (maxIt_ == 0 || iterations < maxIt_)
            && result > minStepLen_)
        {
            // decrease step length
            result *= beta_;
            currLES.construct(state + result * step, errFuncs);
            currVal = currLES.b.norm();
            ++iterations;
        }

        // limit step length by minimum step length
        result = std::max(result, minStepLen_);

        return result;
    }
}
