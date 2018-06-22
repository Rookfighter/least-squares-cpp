/*
 * increasing_line_search.h
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#include "optcpp/increasing_line_search.h"

namespace opt
{
    IncreasingLineSearch::IncreasingLineSearch()
        : LineSearchAlgorithm(), beta_(2.0), maxStepLen_(2.0), minStepLen_(1e-4), maxIt_(0)
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
        // start with minimum step length and then increase
        double currLen = minStepLen_;
        double lastLen = currLen;

        auto errRes = evalErrorFuncs(state, errFuncs);
        double lastErr = squaredError(errRes.val);

        errRes = evalErrorFuncs(state + currLen * step, errFuncs);
        double currErr = squaredError(errRes.val);

        size_t iterations = 0;
        // keep increasing step length while error shows improvement
        while(currErr < lastErr
            && (maxIt_ == 0 || iterations < maxIt_)
            && lastLen < maxStepLen_)
        {
            lastLen = currLen;
            currLen *= beta_;

            errRes = evalErrorFuncs(state + currLen * step, errFuncs);
            lastErr = currErr;
            currErr = squaredError(errRes.val);
        }

        // use las step length as result
        // limit step length by maximum step length
        double result = std::min(lastLen, maxStepLen_);
        return result;
    }
}
