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
        : OptimizationAlgorithm()
    {

    }

    GradientDescent::~GradientDescent()
    {

    }

    double GradientDescent::stepWidth(const Eigen::VectorXd &state,
                                      const Eigen::MatrixXd &jac) const
    {
        double width = 1e-4;
        double fac = 2.0;

        Eigen::VectorXd b = constructEqSys(state, constraints_).b;
        double currLen = b.norm();
        double lastLen = currLen;

        // Increase jump distances
        do
        {
            width *= fac;

            lastLen = currLen;
            currLen = constructEqSys(state - width * jac.transpose() * b,
                                     constraints_).b.norm();
        }
        while(currLen < lastLen);

        // Return last epsilon which minimized
        return width / fac;
    }

    Eigen::VectorXd GradientDescent::calcStepUpdate(const Eigen::VectorXd &state)
    {
        EquationSystem eqRes = constructEqSys(state, constraints_);
        // calculate step width
        double width = stepWidth(state, eqRes.A);
        // TODO limit step width
        width = std::min(width, 2.0);

        // Gradient method
        return width * eqRes.A.transpose() * eqRes.b;
    }
}
