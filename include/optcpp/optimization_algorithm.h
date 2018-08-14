/*
 * optimization_algorithm.h
 *
 *  Created on: 07 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_OPTIMIZATION_ALGORITHM_H_
#define OPT_OPTIMIZATION_ALGORITHM_H_

#include "optcpp/line_search_algorithm.h"
#include <vector>
#include <iostream>

namespace opt
{
    /** Inteface for optimization algorithms. */
    class OptimizationAlgorithm
    {
    protected:
        std::vector<ErrorFunction *> errFuncs_;
        LineSearchAlgorithm *lineSearch_;
        bool verbose_;

        virtual void logStep(const size_t iterations,
            const double error,
            const Eigen::VectorXd &state,
            const Eigen::VectorXd &step,
            const double stepLen) const
        {
            std::cout << "iter=" << iterations
                      << "\terr=" << error
                      << "\tstepLen=" << stepLen
                      << "\tstep=" << step.norm()
                      << "\tstate=[" << state.transpose() << "]" << std::endl;
        }

    public:
        struct Result
        {
            Eigen::VectorXd state;
            double error;
            size_t iterations;
            bool converged;
        };

        OptimizationAlgorithm()
            : errFuncs_(), lineSearch_(nullptr), verbose_(false)
        {}
        OptimizationAlgorithm(const OptimizationAlgorithm &optalg) = delete;
        virtual ~OptimizationAlgorithm()
        {
            if(lineSearch_ != nullptr)
                delete lineSearch_;
            clearErrorFunctions();
        }

        void setVerbose(const bool verbose)
        {
            verbose_ = verbose;
        }

        /** Sets the line search algorithm to determine the step length.
         *  Set nullptr for no line search. The step length is then 1.0.
         *  The line search algorithm is owned by this class.
         *  @param lineSearch line search algorithm */
        void setLineSearchAlgorithm(LineSearchAlgorithm *lineSearch)
        {
            if(lineSearch_ != nullptr)
                delete lineSearch_;
            lineSearch_ = lineSearch;
        }

        /** Sets the error functions to be optimized.
         *  The error functions are owned by this class.
         *  @param errFuncs vector of error functions */
        void setErrorFunctions(const std::vector<ErrorFunction *> &errFuncs)
        {
            clearErrorFunctions();
            errFuncs_ = errFuncs;
        }

        /** Clears and deletes the error functions. */
        void clearErrorFunctions()
        {
            for(ErrorFunction *err : errFuncs_)
                delete err;
            errFuncs_.clear();
        }

        /** Caclculates the step length according the line search algorithm.
         *  @param state current state vector
         *  @param step current optimization step
         *  @return step length */
        double stepLength(
            const Eigen::VectorXd &state, const Eigen::VectorXd &step) const
        {
            if(lineSearch_ == nullptr)
                return 1.0;

            return lineSearch_->stepLength(state, step, errFuncs_);
        }

        /** Calculates the state update vector of the algorithm. The vector
         *  will be added to the state.
         *  @param state current state vector
         *  @param step state update vector
         *  @return squared error */
        virtual Eigen::VectorXd calcStepUpdate(
            const Eigen::VectorXd &state,
            const Eigen::VectorXd &errValue,
            const Eigen::MatrixXd &errJacobian) = 0;

        /** Updates the given state by one step of the algorithm and returns
         *  the new state.
         *  @param state current state vector
         *  @return new state vector */
        Result update(const Eigen::VectorXd &state)
        {
            Result result;
            result.iterations = 1;
            result.converged = true;

            Eigen::VectorXd errValue;
            Eigen::MatrixXd errJacobian;
            evalErrorFuncs(state, errFuncs_, errValue, errJacobian);

            Eigen::VectorXd step = calcStepUpdate(state, errValue, errJacobian);
            double stepLen = stepLength(state, step);

            result.state = state + stepLen * step;

            evalErrorFuncs(result.state, errFuncs_, errValue, errJacobian);
            result.error = squaredError(errValue);

            return result;
        }

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
            const size_t maxIt = 0)
        {
            Result result;
            result.state = state;

            // calculate initial step
            Eigen::VectorXd errValue;
            Eigen::MatrixXd errJacobian;

            // evaluate error functions
            evalErrorFuncs(result.state, errFuncs_, errValue, errJacobian);
            result.error = squaredError(errValue);

            // calculate first state increment
            Eigen::VectorXd step = calcStepUpdate(result.state, errValue, errJacobian);
            double stepLen = stepLength(result.state, step);

            size_t iterations = 0;

            while(step.norm() > eps && (maxIt == 0 || iterations < maxIt))
            {
                // move state
                result.state += stepLen * step;

                // evaluate error functions
                evalErrorFuncs(result.state, errFuncs_, errValue, errJacobian);
                result.error = squaredError(errValue);

                // calculate next state increment
                step = calcStepUpdate(result.state, errValue, errJacobian);
                stepLen = stepLength(result.state, step);

                if(verbose_)
                    logStep(iterations, result.error, result.state, step, stepLen);

                ++iterations;
            }

            result.iterations = iterations;
            result.converged = step.norm() <= eps;

            return result;
        }
    };
}

#endif
