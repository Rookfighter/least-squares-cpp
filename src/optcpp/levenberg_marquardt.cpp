/*
 * levenberg_marquardt.cpp
 *
 *  Created on: 09 May 2018
 *      Author: Fabian Meyer
 */

#include "optcpp/levenberg_marquardt.h"
#include "optcpp/linear_equation_system.h"

namespace opt
{

    LevenbergMarquardt::LevenbergMarquardt()
        : OptimizationAlgorithm(), damping_(1.0), lambda_(1.0), maxIt_(0)
    {

    }

    LevenbergMarquardt::~LevenbergMarquardt()
    {

    }

    void LevenbergMarquardt::setDamping(const double damping)
    {
        damping_ = damping;
    }

    void LevenbergMarquardt::setLambda(const double lambda)
    {
        lambda_ = lambda;
    }

    void LevenbergMarquardt::setMaxIterations(const size_t maxIt)
    {
        maxIt_ = maxIt;
    }

    Eigen::VectorXd LevenbergMarquardt::calcStepUpdate(const Eigen::VectorXd &state)
    {
        LinearEquationSystem eqSysA(state, errFuncs_);
        double errA = eqSysA.b.norm();
        LinearEquationSystem eqSysB;

        Eigen::VectorXd step;
        Eigen::MatrixXd prevA;

        size_t iterations = 0;
        bool found = false;
        while(!found && (maxIt_ == 0 || iterations < maxIt_))
        {
            prevA = eqSysA.A;
            // add gradient descent matrix
            eqSysA.A += lambda_ * Eigen::MatrixXd::Identity(eqSysA.A.rows(),
                        eqSysA.A.cols());
            eqSysA.A *= damping_;

            step = -eqSysA.solveSVD();

            eqSysB.construct(state + step, errFuncs_);
            double errB = eqSysB.b.norm();

            if(errA < errB)
            {
                // new error is greater so don't change state
                // increase lambda
                lambda_ *= 2.0;
                eqSysA.A = std::move(prevA);
            }
            else
            {
                // new error has shown improvement
                // decrease lambda
                lambda_ /= 2.0;
                found = true;
            }

            ++iterations;
        }

        return step;
    }
}
