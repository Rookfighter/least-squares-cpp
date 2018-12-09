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
    template<typename Scalar>
    class GaussNewton : public OptimizationAlgorithm<Scalar>
    {
    private:
        Scalar damping_;
        LinearEquationSystem<Scalar> eqSys_;

    public:
        GaussNewton()
            : OptimizationAlgorithm<Scalar>(), damping_(1.0)
        {}
        GaussNewton(const GaussNewton &gn) = delete;
        ~GaussNewton()
        {}

        void setDamping(const Scalar damping)
        {
            damping_ = damping;
        }

        void computeNewtonStep(
            const Vector<Scalar> &,
            const Vector<Scalar> &errValue,
            const Matrix<Scalar> &errJacobian,
            Vector<Scalar> &outStep) override
        {
            eqSys_.b = errJacobian.transpose() * errValue;
            eqSys_.A = errJacobian.transpose() * errJacobian;

            this->solver_->solve(eqSys_, outStep);
            outStep *= -damping_;
        }
    };
}

#endif
