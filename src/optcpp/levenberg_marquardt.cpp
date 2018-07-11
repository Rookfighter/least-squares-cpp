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
    {}

    LevenbergMarquardt::~LevenbergMarquardt()
    {}

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

    Eigen::VectorXd LevenbergMarquardt::calcStepUpdate(
        const Eigen::VectorXd &state)
    {
        ErrorFunction::Result errResB;
        auto errResA = evalErrorFuncs(state, errFuncs_);
        double errA = squaredError(errResA.val);

        LinearEquationSystem eqSys;
        // set value vector (stays constant)
        eqSys.b = errResA.jac.transpose() * errResA.val;

        Eigen::VectorXd step;
        Eigen::MatrixXd jacSq = errResA.jac.transpose() * errResA.jac;

        size_t iterations = 0;
        bool found = false;
        while(!found && (maxIt_ == 0 || iterations < maxIt_))
        {
            // computer coefficient matrix
            eqSys.A = jacSq;
            eqSys.A += lambda_ * Eigen::MatrixXd::Identity(
                                     eqSys.A.rows(), eqSys.A.cols());

            // solve equation system
            step = -damping_ * eqSys.solveSVD();

            errResB = evalErrorFuncs(state + step, errFuncs_);
            double errB = squaredError(errResB.val);

            if(errA < errB)
            {
                // new error is greater so don't change state
                // increase lambda
                lambda_ *= 2.0;
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
