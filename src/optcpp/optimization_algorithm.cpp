/*
 * optimization_algorithm.cpp
 *
 *  Created on: 07 May 2018
 *      Author: Fabian Meyer
 */

#include <iostream>
#include "optcpp/optimization_algorithm.h"
#include "optcpp/linear_equation_system.h"
#include "optcpp/math.h"

namespace opt
{

    OptimizationAlgorithm::OptimizationAlgorithm()
        : errFuncs_(), lineSearch_(nullptr), verbose_(false)
    {

    }

    OptimizationAlgorithm::~OptimizationAlgorithm()
    {
        if(lineSearch_ != nullptr)
            delete lineSearch_;
        clearErrorFunctions();
    }

    void OptimizationAlgorithm::setVerbose(const bool verbose)
    {
        verbose_ = verbose;
    }

    void OptimizationAlgorithm::setLineSearchAlgorithm(LineSearchAlgorithm
            *lineSearch)
    {
        if(lineSearch_ != nullptr)
            delete lineSearch_;
        lineSearch_ = lineSearch;
    }

    void OptimizationAlgorithm::setErrorFunctions(
        const std::vector<ErrorFunction *> &errFuncs)
    {
        clearErrorFunctions();
        errFuncs_ = errFuncs;
    }

    void OptimizationAlgorithm::clearErrorFunctions()
    {
        for(ErrorFunction *err : errFuncs_)
            delete err;
        errFuncs_.clear();
    }

    double OptimizationAlgorithm::stepLength(
        const Eigen::VectorXd &state,
        const Eigen::VectorXd &step) const
    {
        if(lineSearch_ == nullptr)
            return 1.0;

        return lineSearch_->stepLength(state, step, errFuncs_);
    }

    Eigen::VectorXd OptimizationAlgorithm::update(const Eigen::VectorXd &state)
    {
        Eigen::VectorXd nState(state);

        Eigen::VectorXd step = calcStepUpdate(state);
        double stepLen = stepLength(state, step);

        nState += stepLen * step;

        return nState;
    }

    void OptimizationAlgorithm::logStep(const size_t iterations,
        const Eigen::VectorXd &state,
        const Eigen::VectorXd &step,
        const double stepLen)
    {
        LinearEquationSystem eqSys(state, errFuncs_);
        double err = squaredError(eqSys.b);
        std::cout
            << "iter=" << iterations
            << "\terr=" << err
            << "\tstepLen=" << stepLen
            << "\tstate=[" << state.transpose() << "]"
            << "\tstep=[" <<step.transpose() << "]"
            << std::endl;

    }

    OptimizationAlgorithm::Result OptimizationAlgorithm::run(
        const Eigen::VectorXd &state,
        const double eps,
        const size_t maxIt)
    {
        Result result;
        result.state = state;
        // calculate initial step
        Eigen::VectorXd step = calcStepUpdate(state);
        double stepLen = 1.0;

        size_t iterations = 0;

        while(step.norm() > eps && (maxIt == 0 || iterations < maxIt))
        {
            stepLen = stepLength(state, step);
            result.state += stepLen * step;

            if(verbose_)
                logStep(iterations, result.state, step, stepLen);

            // increment
            step = calcStepUpdate(result.state);
            ++iterations;
        }

        result.iterations = iterations;
        result.converged = step.norm() <= eps;

        return result;
    }
}
