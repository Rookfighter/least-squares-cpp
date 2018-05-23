/*
 * armijo_backtracking.cpp
 *
 *  Created on: 23 May 2018
 *      Author: Fabian Meyer
 */

#include "optcpp/armijo_backtracking.h"

namespace opt
{
    ArmijoBacktracking::ArmijoBacktracking()
    : LineSearchAlgorithm(), beta_(0.8), gamma_(0.1), maxStepLen_(1.0)
    {

    }

    ArmijoBacktracking::~ArmijoBacktracking()
    {

    }

    void ArmijoBacktracking::setBeta(const double beta)
    {
        beta_ = beta;
    }

    void ArmijoBacktracking::setGamma(const double gamma)
    {
        gamma_ = gamma;
    }

    void ArmijoBacktracking::setMaxStepLen(const double stepLen)
    {
        maxStepLen_ = stepLen;
    }

    double ArmijoBacktracking::calcStepLength(
        const Eigen::VectorXd &state,
        const Eigen::VectorXd &step,
        const std::vector<Constraint*> &constraints) const
    {
        double result = maxStepLen_;
        EquationSystem currEqSys = constructEqSys(state + result * step, constraints);
        EquationSystem eqSys = constructEqSys(state, constraints);

        // check for armijo condition
        while(currEqSys.b.norm() >=
             (eqSys.b + gamma_ * result * eqSys.A.transpose() * step).norm())
        {
            // decrease step length
            result *= beta_;

            currEqSys = constructEqSys(state + result * step, constraints);
        }

        return result;
    }
}
