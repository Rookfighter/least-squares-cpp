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

    Eigen::VectorXd solveSVD(const EquationSystem &eqSys)
    {
        Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::FullPivHouseholderQRPreconditioner>
        decomp(eqSys.A.transpose() * eqSys.A,
               Eigen::ComputeFullU | Eigen::ComputeFullV);

        return decomp.solve(eqSys.A.transpose() * eqSys.b);
    }
}
