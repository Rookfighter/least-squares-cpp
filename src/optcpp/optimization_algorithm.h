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

namespace opt
{
    /** Inteface for optimization algorithms. */
    class OptimizationAlgorithm
    {
    protected:
        std::vector<ErrorFunction *> errFuncs_;
        LineSearchAlgorithm *lineSearch_;
    public:
        struct Result
        {
            Eigen::VectorXd state;
            size_t iterations;
            bool converged;
        };

        OptimizationAlgorithm();
        OptimizationAlgorithm(const OptimizationAlgorithm &optalg) = delete;
        virtual ~OptimizationAlgorithm();

        /** Sets the line search algorithm to determine the step length.
         *  Set nullptr for no line search. The step length is then 1.0.
         *  The line search algorithm is owned by this class.
         *  @param lineSearch line search algorithm */
        void setLineSearchAlgorithm(LineSearchAlgorithm *lineSearch);

        /** Sets the error functions to be optimized.
         *  The error functions are owned by this class.
         *  @param errFuncs vector of error functions */
        void setErrorFunctions(const std::vector<ErrorFunction *> &errFuncs);

        /** Clears and deletes the error functions. */
        void clearErrorFunctions();

        /** Caclculates the step length according the line search algorithm.
         *  @param state current state vector
         *  @param step current optimization step
         *  @return step length */
        double stepLength(const Eigen::VectorXd &state,
                          const Eigen::VectorXd &step) const;

        /** Calculates the state update vector of the algorithm. The vector
         *  will be added to the state.
         *  @param state current state vector
         *  @return state update vector */
        virtual Eigen::VectorXd calcStepUpdate(const Eigen::VectorXd &state) = 0;

        /** Updates the given state by one step of the algorithm and returns
         *  the new state.
         *  @param state current state vector
         *  @return new state vector */
        Eigen::VectorXd update(const Eigen::VectorXd &state);

        /** Runs the algorithm on the given initial state. Terminates if either
         *  convergence is achieved or the maximum number of iterations has
         *  been reached.
         *  @param state intial state vector
         *  @param eps epsilon for the convergence condition
         *  @param maxIt maximum number of iterations (0 = infinite)
         *  @return struct with resulting state vector and convergence
         *          information */
        Result run(const Eigen::VectorXd &state,
                   const double eps,
                   const size_t maxIt = 0);
    };
}

#endif
