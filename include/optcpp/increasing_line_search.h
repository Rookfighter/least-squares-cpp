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
    public:
        IncreasingLineSearch()
            : LineSearchAlgorithm(), beta_(2.0)
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

        double search(const Eigen::VectorXd &state,
            const Eigen::VectorXd &step,
            const std::vector<ErrorFunction *> &errFuncs) const override
        {
            // start with minimum step length and then increase
            double currLen = minStepLen_;
            double lastLen = currLen;

            // results of error functions
            Eigen::VectorXd errVal;
            Eigen::MatrixXd errJac;

            evalErrorFuncs(state, errFuncs, errVal, errJac);
            double lastErr = squaredError(errVal);

            evalErrorFuncs(state + currLen * step, errFuncs, errVal, errJac);
            double currErr = squaredError(errVal);

            size_t iterations = 0;
            // keep increasing step length while error shows improvement
            while(currErr < lastErr && (maxIt_ == 0 || iterations < maxIt_) &&
                  lastLen < maxStepLen_)
            {
                lastLen = currLen;
                currLen *= beta_;

                evalErrorFuncs(state + currLen * step, errFuncs, errVal, errJac);
                lastErr = currErr;
                currErr = squaredError(errVal);
            }

            // use last step length as result
            // limit step length by maximum step length
            return std::min(lastLen, maxStepLen_);
        }
    };
}

#endif
