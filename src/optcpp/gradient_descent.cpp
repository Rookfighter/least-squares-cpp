/*
 * gradient_descent.c
 *
 *  Created on: 07 May 2018
 *      Author: Fabian Meyer
 */

#include "optcpp/gradient_descent.h"
#include "optcpp/linear_equation_system.h"

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
        LinearEquationSystem eqSys(state, errFuncs_);
        // Gradient method
        return - (1.0 / damping_) * eqSys.A.transpose() * eqSys.b;
    }
}
