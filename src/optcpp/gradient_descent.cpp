/*
 * gradient_descent.c
 *
 *  Created on: 07 May 2018
 *      Author: Fabian Meyer
 */

#include "optcpp/gradient_descent.h"

namespace opt
{
    GradientDescent::GradientDescent()
        : OptimizationAlgorithm(), damping_(1.0)
    {

    }

    GradientDescent::~GradientDescent()
    {

    }

    void GradientDescent::setDamping(const double damping)
    {
        damping_ = damping;
    }

    Eigen::VectorXd GradientDescent::calcStepUpdate(const Eigen::VectorXd &state)
    {
        auto errRes = evalErrorFuncs(state, errFuncs_);
        // Gradient method
        return -damping_ * errRes.jac.transpose() * errRes.val;
    }
}
