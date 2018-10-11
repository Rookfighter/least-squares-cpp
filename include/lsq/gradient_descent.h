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
    class GradientDescent : public OptimizationAlgorithm
    {
    private:
        double damping_;

    public:
        GradientDescent()
            : OptimizationAlgorithm(), damping_(1.0)
        {}
        GradientDescent(const GradientDescent &gd) = delete;
        ~GradientDescent()
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
            // Gradient method
            outStep = -damping_ * errJacobian.transpose() * errValue;
        }
    };
}

#endif
