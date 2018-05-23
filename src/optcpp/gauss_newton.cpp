/*
 * gauss_newton.cpp
 *
 *  Created on: 03 May 2018
 *      Author: Fabian Meyer
 */

#include "optcpp/gauss_newton.h"

namespace opt
{
    GaussNewton::GaussNewton()
        : OptimizationAlgorithm(), damping_(1.0)
    {

    }

    GaussNewton::~GaussNewton()
    {

    }

    void GaussNewton::setDamping(const double damping)
    {
        damping_ = damping;
    }

    Eigen::VectorXd GaussNewton::calcStepUpdate(const Eigen::VectorXd &state)
    {
        EquationSystem eqSys = constructLEQ(state);
        // damping factor
        eqSys.A *= damping_;
        return solveSVD(eqSys);
    }
}
