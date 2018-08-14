/*
 * line_search_algorithm.h
 *
 *  Created on: 23 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_LINE_SEARCH_ALGORITHM_H_
#define OPT_LINE_SEARCH_ALGORITHM_H_

#include "optcpp/error_function.h"

namespace opt
{
    /** Interface for defining line search algorithms. */
    class LineSearchAlgorithm
    {
    public:
        LineSearchAlgorithm()
        {}
        virtual ~LineSearchAlgorithm()
        {}

        /** Calculates the step length for the next optimization step.
         *  @param state current state vector
         *  @param step the current optimization step
         *  @param errFuncs vector of error functions
         *  @return length of the step */
        virtual double stepLength(const Eigen::VectorXd &state,
            const Eigen::VectorXd &step,
            const std::vector<ErrorFunction *> &errFuncs) const = 0;
    };
}

#endif
