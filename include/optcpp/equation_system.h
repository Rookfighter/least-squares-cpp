/*
 * equation_system.h
 *
 *  Created on: 04 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_EQUATION_SYSTEM_H_
#define OPT_EQUATION_SYSTEM_H_

#include <Eigen/Dense>

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

    Eigen::VectorXd solveSVD(const EquationSystem &eqSys);
}

#endif
