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
    : damping_(1.0)
    {

    }

    LevenbergMarquardt::~LevenbergMarquardt()
    {

    }

    void LevenbergMarquardt::setDamping(const double damping)
    {
        damping_ = damping;
    }

    Eigen::VectorXd LevenbergMarquardt::calcStepUpdate(const Eigen::VectorXd &state)
    {
        EquationSystem eqSys = constructLEQ(state);
        Eigen::VectorXd delta;

        while(true)
        {
            // damping factor
            Eigen::MatrixXd oldA = eqSys.A;
            eqSys.A += damping_ * Eigen::MatrixXd::Identity(eqSys.A.rows(), eqSys.A.cols());
            delta =  solveSVD(eqSys);
            EquationSystem eqSys2 = constructLEQ(state + delta);

            if(eqSys.b.norm() < eqSys2.b.norm())
            {
                // new error is greater so don't change state
                // but increase damping factor
                damping_ *= 2;
                eqSys.A = oldA;
            }
            else
            {
                // new error has shown improvement
                // decrease damping factor
                damping_ /= 2;
                break;
            }
        }
        return delta;
    }
}
