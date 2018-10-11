/*
 * increasing_line_search.h
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#ifndef LSQ_INCREASING_LINE_SEARCH_H_
#define LSQ_INCREASING_LINE_SEARCH_H_

#include "lsq/line_search_algorithm.h"

namespace lsq
{
    /** Implementation of a increasing line search algorithm. */
    class IncreasingLineSearch : public LineSearchAlgorithm
    {
    private:
        double beta_;
        Eigen::VectorXd errVal_;
        Eigen::MatrixXd errJac_;
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
            const std::vector<ErrorFunction *> &errFuncs) override
        {
            // start with minimum step length and then increase
            double currLen = minStepLen_;
            double lastLen = currLen;

            evalErrorFuncs(state, errFuncs, errVal_, errJac_);
            double lastErr = squaredError(errVal_);

            evalErrorFuncs(state + currLen * step, errFuncs, errVal_, errJac_);
            double currErr = squaredError(errVal_);

            size_t iterations = 0;
            // keep increasing step length while error shows improvement
            while(currErr < lastErr && (maxIt_ == 0 || iterations < maxIt_) &&
                  lastLen < maxStepLen_)
            {
                lastLen = currLen;
                currLen *= beta_;

                evalErrorFuncs(state + currLen * step, errFuncs, errVal_, errJac_);
                lastErr = currErr;
                currErr = squaredError(errVal_);
            }

            // use last step length as result
            // limit step length by maximum step length
            return std::min(lastLen, maxStepLen_);
        }
    };
}

#endif
