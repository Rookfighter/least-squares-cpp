/*
 * line_search_algorithm.h
 *
 *  Created on: 23 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_LINE_SEARCH_ALGORITHM_H_
#define OPT_LINE_SEARCH_ALGORITHM_H_

#include "optcpp/equation_system.h"

namespace opt
{
    class LineSearchAlgorithm
    {
    public:
        LineSearchAlgorithm() { }
        virtual ~LineSearchAlgorithm() { }

        virtual double calcStepLength(
            const Eigen::VectorXd &state,
            const Eigen::VectorXd &step,
            const std::vector<Constraint*> &constraints) const = 0;
    };
}

#endif
