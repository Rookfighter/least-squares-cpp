/*
 * levenberg_marquardt.h
 *
 *  Created on: 09 May 2018
 *      Author: Fabian Meyer
 */

#ifndef LSQ_LEVENBERG_MARQUARDT_H_
#define LSQ_LEVENBERG_MARQUARDT_H_

#include "lsq/optimization_algorithm.h"
#include "lsq/linear_equation_system.h"

namespace lsq
{
    /** Implementation of the levelberg marquardt optimization algorithm. */
    class LevenbergMarquardt : public OptimizationAlgorithm
    {
    private:
        double damping_;
        double lambda_;
        size_t maxItLM_;

        Eigen::VectorXd errValB_;
        Eigen::MatrixXd errJacB_;
        Eigen::MatrixXd errJacobianSq_;
        LinearEquationSystem eqSys_;
    public:
        LevenbergMarquardt()
            : OptimizationAlgorithm(), damping_(1.0), lambda_(1.0), maxItLM_(0)
        {}
        LevenbergMarquardt(const LevenbergMarquardt &lm) = delete;
        ~LevenbergMarquardt()
        {}

        void setDamping(const double damping)
        {
            damping_ = damping;
        }

        /** Sets the initial gradient descent factor of levenberg marquardt.
         *  @param lambda gradient descent factor */
        void setLambda(const double lambda)
        {
            lambda_ = lambda;
        }

        /** Sets maximum iterations of the levenberg marquardt optimization.
         *  Set to 0 for infinite iterations.
         *  @param iterations maximum iterations for lambda search */
        void setMaxIterationsLM(const size_t iterations)
        {
            maxItLM_ = iterations;
        }

        void computeNewtonStep(
            const Eigen::VectorXd &state,
            const Eigen::VectorXd &errValue,
            const Eigen::MatrixXd &errJacobian,
            Eigen::VectorXd &outStep) override
        {
            double errB;
            double errA = squaredError(errValue);

            // set value vector (stays constant)
            eqSys_.b = errJacobian.transpose() * errValue;

            errJacobianSq_ = errJacobian.transpose() * errJacobian;

            size_t iterations = 0;
            bool found = false;
            outStep.setZero(state.size());
            while(!found && (maxItLM_ == 0 || iterations < maxItLM_))
            {
                // compute coefficient matrix
                eqSys_.A = errJacobianSq_;
                // add identity matrix
                for(unsigned int i = 0; i < eqSys_.A.rows(); ++i)
                    eqSys_.A(i, i) += lambda_;

                // solve equation system
                outStep = -damping_ * eqSys_.solveSVD();

                evalErrorFuncs(state + outStep, errFuncs_, errValB_, errJacB_);
                errB = squaredError(errValB_);

                if(errA < errB)
                {
                    // new error is greater so don't change state
                    // increase lambda
                    lambda_ *= 2.0;
                }
                else
                {
                    // new error has shown improvement
                    // decrease lambda
                    lambda_ /= 2.0;
                    found = true;
                }

                ++iterations;
            }
        }
    };
}

#endif
