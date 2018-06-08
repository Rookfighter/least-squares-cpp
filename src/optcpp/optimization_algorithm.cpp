/*
 * optimization_algorithm.cpp
 *
 *  Created on: 07 May 2018
 *      Author: Fabian Meyer
 */

#include "optcpp/optimization_algorithm.h"

namespace opt
{

    OptimizationAlgorithm::OptimizationAlgorithm()
        : constraints_(), lineSearch_(nullptr)
    {

    }

    OptimizationAlgorithm::~OptimizationAlgorithm()
    {
        if(lineSearch_ != nullptr)
            delete lineSearch_;
    }

    void OptimizationAlgorithm::setLineSearchAlgorithm(LineSearchAlgorithm
            *lineSearch)
    {
        if(lineSearch_ != nullptr)
            delete lineSearch_;
        lineSearch_ = lineSearch;
    }

    void OptimizationAlgorithm::setConstraints(const std::vector<Constraint *>
            &constraints)
    {
        constraints_ = constraints;
    }

    void OptimizationAlgorithm::clearConstraints()
    {
        constraints_.clear();
    }

    double OptimizationAlgorithm::calcStepLength(const Eigen::VectorXd &state,
            const Eigen::VectorXd &step) const
    {
        if(lineSearch_ == nullptr)
            return 1.0;

        return lineSearch_->calcStepLength(state, step, constraints_);
    }

    Eigen::VectorXd OptimizationAlgorithm::update(const Eigen::VectorXd &state)
    {
        Eigen::VectorXd nState(state);

        Eigen::VectorXd step = calcStepUpdate(state);
        double stepLen = calcStepLength(state, step);

        nState -= stepLen * step;

        return nState;
    }

    OptimizationAlgorithm::Result OptimizationAlgorithm::run(
        const Eigen::VectorXd &state,
        const double eps,
        const unsigned int maxSteps)
    {
        Result result;
        result.state = state;
        Eigen::VectorXd stepUpdate = calcStepUpdate(state);

        unsigned int step = 0;

        while(stepUpdate.norm() > eps && (maxSteps == 0 || step < maxSteps))
        {
            result.state -= stepUpdate;
            stepUpdate = calcStepUpdate(result.state);
            ++step;
        }

        result.steps = step;
        result.converged = stepUpdate.norm() > eps;

        return result;
    }
}
