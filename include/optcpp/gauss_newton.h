/*
 * gauss_newton.h
 *
 *  Created on: 03 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_GAUSS_NEWTON_H_
#define OPT_GAUSS_NEWTON_H_

#include "optcpp/optimization_algorithm.h"
#include "optcpp/linear_equation_system.h"

namespace opt
{
    /** Implementation of the gauss newton optimization algorithm. */
    class GaussNewton : public OptimizationAlgorithm
    {
    private:
        double damping_;

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

        Eigen::VectorXd calcStepUpdate(
            const Eigen::VectorXd &,
            const Eigen::VectorXd &errValue,
            const Eigen::MatrixXd &errJacobian) override
        {
            LinearEquationSystem eqSys;
            eqSys.b = errJacobian.transpose() * errValue;
            eqSys.A = errJacobian.transpose() * errJacobian;

            return -damping_ * eqSys.solveSVD();
        }
    };
}

#endif
