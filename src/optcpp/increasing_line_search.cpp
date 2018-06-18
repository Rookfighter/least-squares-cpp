/*
 * increasing_line_search.h
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#include "optcpp/increasing_line_search.h"
#include "optcpp/linear_equation_system.h"

namespace opt
{
    IncreasingLineSearch::IncreasingLineSearch()
        : beta_(2.0), maxStepLen_(2.0), minStepLen_(1e-4), maxIt_(0)
    {

    }

    IncreasingLineSearch::~IncreasingLineSearch()
    {

    }

    void IncreasingLineSearch::setBeta(const double beta)
    {
        assert(beta_ > 1.0);
        beta_ = beta;
    }

    void IncreasingLineSearch::setBounds(const double minLen, const double maxLen)
    {
        assert(minStepLen_ < maxStepLen_);

        maxStepLen_ = maxLen;
        minStepLen_ = minLen;
    }

    void IncreasingLineSearch::setMaxIterations(const size_t maxIt)
    {
        maxIt_ = maxIt;
    }

    double IncreasingLineSearch::stepLength(
        const Eigen::VectorXd &state,
        const Eigen::VectorXd &step,
        const std::vector<ErrorFunction *> &errFuncs) const
    {
        double currLen = minStepLen_;
        double lastLen = currLen;

        LinearEquationSystem eqSys(state, errFuncs);
        double lastErr = eqSys.b.norm();

        eqSys.construct(state + currLen * step, errFuncs);
        double currErr = eqSys.b.norm();

        size_t iterations = 0;
        while(currErr < lastErr && (maxIt_ == 0 || iterations < maxIt_))
        {
            lastLen = currLen;
            currLen *= beta_;

            eqSys.construct(state + currLen * step, errFuncs);
            lastErr = currErr;
            currErr = eqSys.b.norm();
        }

        // use las step length as result
        // limit step length by maximum step length
        double result = std::min(lastLen, maxStepLen_);
        return result;
    }
}
