/*
 * equation_system.h
 *
 *  Created on: 04 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_EQUATION_SYSTEM_H_
#define OPT_EQUATION_SYSTEM_H_

#include <Eigen/Dense>
#include "optcpp/constraint.h"

namespace opt
{
    struct EquationSystem
    {
        /** Function vector. */
        Eigen::VectorXd b;
        /** Jacobi matrix. */
        Eigen::MatrixXd A;

        unsigned int equations() const;
        unsigned int unknowns() const;
        bool undertermined() const;
    };

    /**
     * Constructs a linear equation system from the constraints given the
     * current state.
     * @return linear equation system
     */
    EquationSystem constructEqSys(const Eigen::VectorXd &state, const std::vector<Constraint*> &constraints);

    Eigen::VectorXd solveSVD(const EquationSystem &eqSys);
}

#endif
