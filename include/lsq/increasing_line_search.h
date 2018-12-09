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
    template<typename Scalar>
    class IncreasingLineSearch : public LineSearchAlgorithm<Scalar>
    {
    private:
        Scalar beta_;
        Vector<Scalar> errVal_;
        Matrix<Scalar> errJac_;
    public:
        IncreasingLineSearch()
            : LineSearchAlgorithm<Scalar>(), beta_(2.0)
        {}
        ~IncreasingLineSearch()
        {}

        /** Sets the increasing factor during step calculation. The value must
         *  be in the interval (1 inf). Choose not too big, e.g. 2.0.
         *  @param beta increasing factor */
        void setBeta(const Scalar beta)
        {
            assert(beta_ > 1.0);
            beta_ = beta;
        }

        double search(const Vector<Scalar> &state,
            const Vector<Scalar> &step,
            ErrorFunction<Scalar> &errFunc) override
        {
            // start with minimum step length and then increase
            Scalar currLen = this->minStepLen_;
            Scalar lastLen = currLen;

            errFunc.evaluate(state, errVal_, errJac_);
            Scalar lastErr = squaredError<Scalar>(errVal_);

            errFunc.evaluate(state + currLen * step, errVal_, errJac_);
            Scalar currErr = squaredError(errVal_);

            size_t iterations = 0;
            // keep increasing step length while error shows improvement
            while(currErr < lastErr && (this->maxIt_ == 0 ||
                  iterations < this->maxIt_) &&
                  lastLen < this->maxStepLen_)
            {
                lastLen = currLen;
                currLen *= beta_;

                errFunc.evaluate(state + currLen * step, errVal_, errJac_);
                lastErr = currErr;
                currErr = squaredError<Scalar>(errVal_);
            }

            // use last step length as result
            // limit step length by maximum step length
            return std::min<Scalar>(lastLen, this->maxStepLen_);
        }
    };
}

#endif
