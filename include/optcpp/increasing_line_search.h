/*
 * increasing_line_search.h
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_INCREASING_LINE_SEARCH_H_
#define OPT_INCREASING_LINE_SEARCH_H_

#include "optcpp/line_search_algorithm.h"

namespace opt
{
    /** Implementation of a increasing line search algorithm. */
    class IncreasingLineSearch : public LineSearchAlgorithm
    {
    private:
        double beta_;
        double maxStepLen_;
        double minStepLen_;
        size_t maxIt_;

    public:
        IncreasingLineSearch()
            : LineSearchAlgorithm(), beta_(2.0), maxStepLen_(2.0),
            minStepLen_(1e-4), maxIt_(0)
        {}
        ~IncreasingLineSearch()
        {}

        /** Sets the increasing factor during step calculation. The value must
         *  be in the interval (1 inf). Choose not too big, e.g. 2.0.
         *  @param beta increasing factor */
        void setBeta(const double beta)
        {
            assert(beta_ > 1.0);
            beta_ = beta;
        }

        /** Sets the bounds for the step length. The step length is then
         *  assured to be in the interval [minLen, maxLen].
         *  @param minLen minimum step length
         *  @param maxLen maximum step length */
        void setBounds(const double minLen, const double maxLen)
        {
            assert(minStepLen_ < maxStepLen_);

            maxStepLen_ = maxLen;
            minStepLen_ = minLen;
        }

        /** Sets maximum iterations for the line search.
         *  Set to 0 for infinite iterations.
         *  @param maxIt maximum iterations */
        void setMaxIterations(const size_t maxIt)
        {
            maxIt_ = maxIt;
        }

        double stepLength(const Eigen::VectorXd &state,
            const Eigen::VectorXd &step,
            const std::vector<ErrorFunction *> &errFuncs) const override
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
            while(currErr < lastErr && (maxIt_ == 0 || iterations < maxIt_) &&
                  lastLen < maxStepLen_)
            {
                lastLen = currLen;
                currLen *= beta_;

                errRes = evalErrorFuncs(state + currLen * step, errFuncs);
                lastErr = currErr;
                currErr = squaredError(errRes.val);
            }

            // use last step length as result
            // limit step length by maximum step length
            double result = std::min(lastLen, maxStepLen_);
            return result;
        }
    };
}

#endif
