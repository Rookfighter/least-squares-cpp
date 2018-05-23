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
    : OptimizationAlgorithm()
    {

    }

    GaussNewton::~GaussNewton()
    {

    }

    Eigen::VectorXd GaussNewton::calcStepUpdate(const Eigen::VectorXd &state)
    {
        EquationSystem eqSys = constructLEQ(state);
        assert(!eqSys.undertermined());
        // damping factor
        eqSys.A *= 4.0;
        return solveSVD(eqSys);
    }
}
