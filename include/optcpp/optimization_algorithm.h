/*
 * optimization_algorithm.h
 *
 *  Created on: 07 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_OPTIMIZATION_ALGORITHM_H_
#define OPT_OPTIMIZATION_ALGORITHM_H_

#include <vector>
#include "optcpp/line_search_algorithm.h"
#include "optcpp/constraint.h"

namespace opt
{
    class OptimizationAlgorithm
    {
    protected:
        std::vector<Constraint *> constraints_;
        LineSearchAlgorithm *lineSearch_;
    public:
        struct Result
        {
            Eigen::VectorXd state;
            unsigned int steps;
            bool converged;
        };

        OptimizationAlgorithm();
        virtual ~OptimizationAlgorithm();

        void setLineSearchAlgorithm(LineSearchAlgorithm *lineSearch);
        void setConstraints(const std::vector<Constraint *> &constraints);
        void clearConstraints();

        double calcStepLength(const Eigen::VectorXd &state,
                              const Eigen::VectorXd &step) const;

        /**
         * Calculates the state update vector of the algorithm. The vector will be subtracted
         * from the state.
         * @param state current state vector
         * @return state update vector
         */
        virtual Eigen::VectorXd calcStepUpdate(const Eigen::VectorXd &state) = 0;

        /**
         * Updates the given state by one step of the algorithm and returns the new state.
         * @param state current state vector
         * @return new state vector
         */
        Eigen::VectorXd update(const Eigen::VectorXd &state);

        /**
         * Runs the algorithm on the given initial state. Terminates if either
         * convergence is achieved or the maximum number of iterations has been
         * reached.
         * @param state intial state vector
         * @param eps epsilon for the convergence condition
         * @param maxSteps maximum number of iterations (0 = infinite)
         * @return resulting state vector
         */
        Result run(const Eigen::VectorXd &state, const double eps,
                   const unsigned int maxSteps = 0);
    };
}

#endif
