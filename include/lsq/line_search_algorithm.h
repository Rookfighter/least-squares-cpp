/*
 * line_search_algorithm.h
 *
 *  Created on: 23 May 2018
 *      Author: Fabian Meyer
 */

#ifndef LSQ_LINE_SEARCH_ALGORITHM_H_
#define LSQ_LINE_SEARCH_ALGORITHM_H_

#include "lsq/error_function.h"

namespace lsq
{
    /** Interface for defining line search algorithms. */
    template<typename Scalar>
    class LineSearchAlgorithm
    {
    protected:
        bool verbose_;
        size_t maxIt_;
        Scalar maxStepLen_;
        Scalar minStepLen_;
    public:
        LineSearchAlgorithm()
            : verbose_(false), maxIt_(0), maxStepLen_(1), minStepLen_(1e-6)
        {}
        virtual ~LineSearchAlgorithm()
        {}

        /** Sets the maximum iterations for the line search.
         *  Set to 0 for infinite iterations.
         *  @param iterations maximum iterations */
        void setMaxIterations(const size_t iterations)
        {
            maxIt_ = iterations;
        }

        /** Set verbosity of the algorithm.
         *  If set to true the algorithm writes information about each
         *  iteration on stdout.
         *  @param verbose enable/disable verbosity */
        void setVerbose(const bool verbose)
        {
            verbose_ = verbose;
        }

        /** Sets the bounds for the step length. The step length is then
         *  assured to be in the interval [minLen, maxLen].
         *  @param minLen minimum step length
         *  @param maxLen maximum step length */
        void setBounds(const Scalar minLen, const Scalar maxLen)
        {
            assert(minLen < maxLen);

            maxStepLen_ = maxLen;
            minStepLen_ = minLen;
        }

        /** Calculates the step length for the next optimization step.
         *  Searches along a line on the newton step for the most suitable
         *  length.
         *  @param state current state vector
         *  @param step the current optimization step
         *  @param errFuncs vector of error functions
         *  @return length of the step */
        virtual double search(const Vector<Scalar> &state,
            const Vector<Scalar> &step,
            ErrorFunction<Scalar> &errFunc) = 0;
    };
}

#endif
