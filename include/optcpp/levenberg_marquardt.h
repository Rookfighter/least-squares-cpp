/*
 * levenberg_marquardt.h
 *
 *  Created on: 09 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_LEVENBERG_MARQUARDT_H_
#define OPT_LEVENBERG_MARQUARDT_H_

#include "optcpp/optimization_algorithm.h"
#include "optcpp/linear_equation_system.h"

namespace opt
{
    /** Implementation of the levelberg marquardt optimization algorithm. */
    class LevenbergMarquardt : public OptimizationAlgorithm
    {
    private:
        double damping_;
        double lambda_;
        size_t maxIt_;

    public:
        LevenbergMarquardt()
            : OptimizationAlgorithm(), damping_(1.0), lambda_(1.0), maxIt_(0)
        {}
        LevenbergMarquardt(const LevenbergMarquardt &lm) = delete;
        ~LevenbergMarquardt()
        {}

        void setDamping(const double damping)
        {
            damping_ = damping;
        }

        /** Sets the gradient descent factor of levenberg marquardt.
         *  @param lambda gradient descent factor */
        void setLambda(const double lambda)
        {
            lambda_ = lambda;
        }

        /** Sets maximum iterations of the levenberg marquardt optimization.
         *  Set to 0 for infinite iterations.
         *  @param maxIt maximum iteration for optimization */
        void setMaxIterations(const size_t maxIt)
        {
            maxIt_ = maxIt;
        }

        double calcStepUpdate(const Eigen::VectorXd &state,
            Eigen::VectorXd &step) override
        {
            Eigen::VectorXd errValA;
            Eigen::MatrixXd errJacA;
            Eigen::VectorXd errValB;
            Eigen::MatrixXd errJacB;
            double errB;

            evalErrorFuncs(state, errFuncs_, errValA, errJacA);
            double errA = squaredError(errValA);

            LinearEquationSystem eqSys;
            // set value vector (stays constant)
            eqSys.b = errJacA.transpose() * errValA;

            Eigen::MatrixXd jacSq = errJacA.transpose() * errJacA;

            size_t iterations = 0;
            bool found = false;
            while(!found && (maxIt_ == 0 || iterations < maxIt_))
            {
                // computer coefficient matrix
                eqSys.A = jacSq;
                eqSys.A += lambda_ * Eigen::MatrixXd::Identity(
                                         eqSys.A.rows(), eqSys.A.cols());

                // solve equation system
                step = -damping_ * eqSys.solveSVD();

                evalErrorFuncs(state + step, errFuncs_, errValB, errJacB);
                errB = squaredError(errValB);

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

            return errB;
        }
    };
}

#endif
