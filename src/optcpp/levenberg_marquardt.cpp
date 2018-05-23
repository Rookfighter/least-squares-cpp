/*
 * levenberg_marquardt.cpp
 *
 *  Created on: 09 May 2018
 *      Author: Fabian Meyer
 */

#include "optcpp/levenberg_marquardt.h"

namespace opt
{

    LevenbergMarquardt::LevenbergMarquardt()
    : OptimizationAlgorithm(), damping_(1.0), gradientFac_(1.0)
    {

    }

    LevenbergMarquardt::~LevenbergMarquardt()
    {

    }

    void LevenbergMarquardt::setDamping(const double damping)
    {
        damping_ = damping;
    }

    void LevenbergMarquardt::setGradientFactor(const double fac)
    {
        gradientFac_ = fac;
    }

    Eigen::VectorXd LevenbergMarquardt::calcStepUpdate(const Eigen::VectorXd &state)
    {
        EquationSystem eqSys = constructEqSys(state, constraints_);
        Eigen::VectorXd delta;

        while(true)
        {
            Eigen::MatrixXd oldA = eqSys.A;
            eqSys.A += gradientFac_ * Eigen::MatrixXd::Identity(eqSys.A.rows(), eqSys.A.cols());
            eqSys.A *= damping_;
            delta =  solveSVD(eqSys);
            EquationSystem eqSys2 = constructEqSys(state + delta, constraints_);

            if(eqSys.b.norm() < eqSys2.b.norm())
            {
                // new error is greater so don't change state
                // but increase gradient factor
                gradientFac_ *= 2;
                eqSys.A = oldA;
            }
            else
            {
                // new error has shown improvement
                // decrease gradient factor
                gradientFac_ /= 2;
                break;
            }
        }
        return delta;
    }
}
