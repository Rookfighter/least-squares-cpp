/*
 * optimization_algorithm.cpp
 *
 *  Created on: 07 May 2018
 *      Author: Fabian Meyer
 */

#include "optcpp/optimization_algorithm.h"

namespace opt
{
    static unsigned int equations(const std::vector<Constraint*> &constraints)
    {
        unsigned int sum = 0;
        for(const Constraint *c :constraints)
            sum += c->outputSize();
        return sum;
    }

    static unsigned int unknowns(const Eigen::VectorXd &state)
    {
        return state.size();
    }

    OptimizationAlgorithm::OptimizationAlgorithm()
    : constraints_()
    {

    }

    OptimizationAlgorithm::~OptimizationAlgorithm()
    {

    }

    void OptimizationAlgorithm::setConstraints(const std::vector<Constraint*> &constraints)
    {
        constraints_ = constraints;
    }

    void OptimizationAlgorithm::clearConstraints()
    {
        constraints_.clear();
    }

    EquationSystem OptimizationAlgorithm::constructLEQ(const Eigen::VectorXd &state) const
    {
        EquationSystem result;
        result.b.setZero(equations(constraints_));
        result.A.setZero(equations(constraints_), unknowns(state));

        // keep track of the constraint index since constraints can
        // return arbitrary amount of values
        unsigned int cidx = 0;
        for(unsigned int i = 0; i < constraints_.size(); ++i)
        {
            Constraint *constr = constraints_[i];

            // calculate error function of the current constraint
            Constraint::Result funcResult = constr->errorFunc(state);
            for(unsigned int j = 0; j < funcResult.val.size(); ++j)
                result.b(cidx + j) = funcResult.val(j);

            // copy whole jacobian into one row of jacobi matrix
            for(unsigned int row = 0; row < funcResult.jac.rows(); ++row)
            {
                for(unsigned int col = 0; col < funcResult.jac.cols(); ++col)
                    result.A(cidx + row, col) = funcResult.jac(row, col);
            }

            cidx += constr->outputSize();
        }

        return result;
    }


    Eigen::VectorXd OptimizationAlgorithm::update(const Eigen::VectorXd &state)
    {
        Eigen::VectorXd nState(state);

        nState -= calcStepUpdate(state);

        return nState;
    }

    OptimizationAlgorithm::Result OptimizationAlgorithm::run(const Eigen::VectorXd &state,
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
