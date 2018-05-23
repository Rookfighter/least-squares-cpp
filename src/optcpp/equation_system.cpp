/*
 * equation_system.cpp
 *
 *  Created on: 04 May 2018
 *      Author: Fabian Meyer
 */

#include "optcpp/equation_system.h"

namespace opt
{

    unsigned int EquationSystem::equations() const
    {
        return b.size();
    }

    unsigned int EquationSystem::unknowns() const
    {
        return A.cols();
    }

    bool EquationSystem::undertermined() const
    {
        return unknowns() - equations() > 0;
    }

    static unsigned int equations(const std::vector<Constraint *> &constraints)
    {
        unsigned int sum = 0;
        for(const Constraint *c : constraints)
            sum += c->outputSize();
        return sum;
    }

    static unsigned int unknowns(const Eigen::VectorXd &state)
    {
        return state.size();
    }

    EquationSystem constructEqSys(const Eigen::VectorXd &state, const std::vector<Constraint*> &constraints)
    {
        EquationSystem result;
        result.b.setZero(equations(constraints));
        result.A.setZero(equations(constraints), unknowns(state));

        // keep track of the constraint index since constraints can
        // return arbitrary amount of values
        unsigned int cidx = 0;
        for(unsigned int i = 0; i < constraints.size(); ++i)
        {
            Constraint *constr = constraints[i];

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

    Eigen::VectorXd solveSVD(const EquationSystem &eqSys)
    {
        Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::FullPivHouseholderQRPreconditioner>
        decomp(eqSys.A.transpose() * eqSys.A,
               Eigen::ComputeFullU | Eigen::ComputeFullV);

        return decomp.solve(eqSys.A.transpose() * eqSys.b);
    }
}
