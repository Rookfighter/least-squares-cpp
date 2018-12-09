/*
 * gradient_descent.h
 *
 *  Created on: 07 May 2018
 *      Author: Fabian Meyer
 */

#ifndef LSQ_GRADIENT_DESCENT_H_
#define LSQ_GRADIENT_DESCENT_H_

#include "lsq/optimization_algorithm.h"

namespace lsq
{
    /** Implementation of the gradient descent optimization algorithm. */
    template<typename Scalar>
    class GradientDescent : public OptimizationAlgorithm<Scalar>
    {
    private:
        Scalar damping_;

    public:
        GradientDescent()
            : OptimizationAlgorithm<Scalar>(), damping_(1)
        {}
        GradientDescent(const GradientDescent &gd) = delete;
        ~GradientDescent()
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
            // Gradient method
            outStep = -damping_ * errJacobian.transpose() * errValue;
        }
    };
}

#endif
