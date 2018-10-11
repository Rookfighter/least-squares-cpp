/*
 * gauss_newton.h
 *
 *  Created on: 03 May 2018
 *      Author: Fabian Meyer
 */

#ifndef LSQ_GAUSS_NEWTON_H_
#define LSQ_GAUSS_NEWTON_H_

#include "lsq/optimization_algorithm.h"
#include "lsq/linear_equation_system.h"

namespace lsq
{
    /** Implementation of the gauss newton optimization algorithm. */
    class GaussNewton : public OptimizationAlgorithm
    {
    private:
        double damping_;
        LinearEquationSystem eqSys_;

    public:
        GaussNewton()
            : OptimizationAlgorithm(), damping_(1.0)
        {}
        GaussNewton(const GaussNewton &gn) = delete;
        ~GaussNewton()
        {}

        void setDamping(const double damping)
        {
            damping_ = damping;
        }

        void computeNewtonStep(
            const Eigen::VectorXd &,
            const Eigen::VectorXd &errValue,
            const Eigen::MatrixXd &errJacobian,
            Eigen::VectorXd &outStep) override
        {
            eqSys_.b = errJacobian.transpose() * errValue;
            eqSys_.A = errJacobian.transpose() * errJacobian;

            outStep = -damping_ * eqSys_.solveSVD();
        }
    };
}

#endif
