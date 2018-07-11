/*
 * linear_equation_system.cpp
 *
 *  Created on: 04 May 2018
 *      Author: Fabian Meyer
 */

#include "optcpp/linear_equation_system.h"

namespace opt
{
    LinearEquationSystem::LinearEquationSystem() : b(), A()
    {}

    LinearEquationSystem::LinearEquationSystem(
        const Eigen::VectorXd &b, const Eigen::MatrixXd &A)
        : b(b), A(A)
    {}

    LinearEquationSystem::~LinearEquationSystem()
    {}

    size_t LinearEquationSystem::equations() const
    {
        return b.size();
    }

    size_t LinearEquationSystem::unknowns() const
    {
        return A.cols();
    }

    bool LinearEquationSystem::underdetermined() const
    {
        return unknowns() - equations() > 0;
    }

    Eigen::VectorXd LinearEquationSystem::solveSVD() const
    {
        Eigen::JacobiSVD<Eigen::MatrixXd,
            Eigen::FullPivHouseholderQRPreconditioner>
            decomp(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

        return decomp.solve(b);
    }
}
