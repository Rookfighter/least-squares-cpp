/*
 * linear_equation_system.h
 *
 *  Created on: 04 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_LINEAR_EQUATION_SYSTEM_H_
#define OPT_LINEAR_EQUATION_SYSTEM_H_

#include "optcpp/error_function.h"

namespace opt
{
    /** Class representing a linear equation system. Provides functions to
     *  construct the system from error functions and solve it. */
    class LinearEquationSystem
    {
    public:
        /** Constant vector. */
        Eigen::VectorXd b;
        /** Coefficient matrix. */
        Eigen::MatrixXd A;

        LinearEquationSystem();
        LinearEquationSystem(
            const Eigen::VectorXd &b, const Eigen::MatrixXd &A);
        ~LinearEquationSystem();

        /** Solves the linear equation system using SVD decomposition.
         *  @return */
        Eigen::VectorXd solveSVD() const;

        /** Returns the amount of equations provided by this system.
         *  Only valid after A and b have been set appropriately.
         *  @return number of equations */
        size_t equations() const;

        /** Returns the amount of unknowns in this system.
         *  Only valid after A and b have been set appropriately.
         *  @return number of unknowns */
        size_t unknowns() const;

        /** Checks if the system is underdetermined, i.e. equations < unknowns.
         *  Only valid after A and b have been set appropriately.
         *  @return true if the system is undertermined, else false */
        bool underdetermined() const;
    };
}

#endif
