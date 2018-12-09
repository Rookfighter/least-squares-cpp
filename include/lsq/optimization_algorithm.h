/*
 * optimization_algorithm.h
 *
 *  Created on: 07 May 2018
 *      Author: Fabian Meyer
 */

#ifndef LSQ_OPTIMIZATION_ALGORITHM_H_
#define LSQ_OPTIMIZATION_ALGORITHM_H_

#include "lsq/line_search_algorithm.h"
#include "lsq/solver_dense_svd.h"
#include <vector>
#include <iostream>

namespace lsq
{
    /** Inteface for optimization algorithms. */
    template<typename Scalar>
    class OptimizationAlgorithm
    {
    protected:
        ErrorFunction<Scalar> *errorFunc_;
        LineSearchAlgorithm<Scalar> *lineSearch_;
        Solver<Scalar> *solver_;

        bool verbose_;
        size_t maxIt_;
        Scalar eps_;

        virtual void logStep(const size_t iterations,
            const double error,
            const Vector<Scalar> &state,
            const Vector<Scalar> &step,
            const double stepLen) const
        {
            std::cout << "iter=" << iterations
                      << "\terr=" << error
                      << "\tstepLen=" << stepLen
                      << "\tstep=" << step.norm()
                      << "\tstate=[" << state.transpose() << "]" << std::endl;
        }

        void calcStep(
            const Vector<Scalar> &state,
            Vector<Scalar> &outValue,
            Matrix<Scalar> &outJacobian,
            Vector<Scalar> &outStep)
        {
            // evaluate error functions
            errFunc_->evaluate(state, outValue, outJacobian);
            computeNewtonStep(state, outValue, outJacobian, outStep);
        }


    public:
        struct Result
        {
            Vector<Scalar> state;
            Scalar error;
            size_t iterations;
            bool converged;

            Result()
                : state(), error(0), iterations(0), converged(false)
            {}
        };

        OptimizationAlgorithm()
            : errFunc_(nullptr), lineSearch_(nullptr),
            solver_(new SolverDenseSVD()), verbose_(false), maxIt_(0),
            eps_(1e-6)
        {}
        OptimizationAlgorithm(const OptimizationAlgorithm &optalg) = delete;
        virtual ~OptimizationAlgorithm()
        {
            if(lineSearch_ != nullptr)
                delete lineSearch_;
            if(solver_ != nullptr)
                delete solver_;
            if(errFunc_ != nullptr)
                delete errFunc;
        }

        /** Set verbosity of the algorithm.
         *  If set to true the algorithm writes information about each
         *  iteration on stdout.
         *  @param verbose enable/disable verbosity */
        void setVerbose(const bool verbose)
        {
            verbose_ = verbose;
        }

        /** Sets maximum iterations for the optimization process.
         *  Set to 0 for infinite iterations. If the algorithm reaches the
         *  maximum iterations it terminates and returns as "not converged".
         *  @param iterations maximum iterations */
        void setMaxIterations(const size_t iterations)
        {
            maxIt_ = iterations;
        }

        /** Set the convergence criterion of the optimization.
         *  If the length of incremental newton step of the optimization is
         *  less than eps, the algorithm will stop and return as "converged".
         *  @param eps epsilon of the convergence criterion */
        void setEpsilon(const double eps)
        {
            eps_ = eps;
        }

        /** Sets the line search algorithm to determine the step length.
         *  Set nullptr for no line search. The step length is then 1.0.
         *  The line search algorithm is owned by this class.
         *  @param lineSearch line search algorithm */
        void setLineSearchAlgorithm(LineSearchAlgorithm<Scalar> *lineSearch)
        {
            if(lineSearch_ != nullptr)
                delete lineSearch_;
            lineSearch_ = lineSearch;
        }

        /** Sets the solver to solve linear equation systems.
         *  @param solver linear equation system solver */
        void setSolver(Solver<Scalar> *solver)
        {
            if(solver_ != nullptr)
                delete solver_;
            solver_ = solver;
        }

        /** Sets the error function to be optimized.
         *  The error function is owned by this class.
         *  @param errFunc error function */
        void setErrorFunction(ErrorFunction<Scalar> *errFunc)
        {
            if(errFunc_ != nullptr)
                delete errFunc_
            errFunc_ = errFunc;
        }

        /** Caclculates the step length according to the line search algorithm.
         *  @param state current state vector
         *  @param step current optimization step
         *  @return step length */
        Scalar performLineSearch(const Vector<Scalar> &state,
            const Vector<Scalar> &step)
        {
            if(lineSearch_ == nullptr)
                return 1.0;

            return lineSearch_->search(state, step, errFuncs_);
        }

        /** Calculates the state update vector of the algorithm. The vector
         *  will be added to the state.
         *  @param state current state vector
         *  @param errValue values of the error functions of the current state
         *  @param errJacobian jacobian of the error functions of the current
         *         state
         *  @param outStep step state update vector */
        virtual void computeNewtonStep(
            const Vector<Scalar> &state,
            const Vector<Scalar> &errValue,
            const Matrix<Scalar> &errJacobian,
            Vector<Scalar> &outStep) = 0;

        /** Runs the algorithm on the given initial state. Terminates if either
         *  convergence is achieved or the maximum number of iterations has
         *  been reached.
         *  @param state intial state vector
         *  @return struct with resulting state vector and convergence
         *          information */
        Result optimize(const Vector<Scalar> &state)
        {
            Result result;
            result.state = state;

            // init optimization vectors
            Vector<Scalar> errValue;
            Matrix<Scalar> errJacobian;
            Vector<Scalar> step;
            Vector<Scalar> scaledStep;

            // calculate first state increment
            calcStep(result.state, errValue, errJacobian, step);
            Scalar stepLen = performLineSearch(result.state, step);
            scaledStep = stepLen * step;
            result.error = squaredError<Scalar>(errValue);

            size_t iterations = 0;

            while(scaledStep.norm() > eps_ && (maxIt_ == 0 || iterations < maxIt_))
            {
                // move state
                result.state += scaledStep;

                // calculate next state increment
                calcStep(result.state, errValue, errJacobian, step);
                stepLen = performLineSearch(result.state, step);
                scaledStep = stepLen * step;
                result.error = squaredError(errValue);

                if(verbose_)
                    logStep(iterations, result.error, result.state, step, stepLen);

                ++iterations;
            }

            result.iterations = iterations;
            result.converged = scaledStep.norm() <= eps_;

            return result;
        }
    };
}

#endif
