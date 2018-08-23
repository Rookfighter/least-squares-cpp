/*
 * linear_equation_system.h
 *
 *  Created on: 04 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_LINEAR_EQUATION_SYSTEM_H_
#define OPT_LINEAR_EQUATION_SYSTEM_H_

#include <Eigen/Geometry>

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

        LinearEquationSystem()
        : b(), A()
        {}

        LinearEquationSystem(
            const Eigen::VectorXd &b, const Eigen::MatrixXd &A)
            : b(b), A(A)
        {}

        /** Solves the linear equation system using SVD decomposition.
         *  @return */
        Eigen::VectorXd solveSVD() const
        {
            Eigen::JacobiSVD<Eigen::MatrixXd,
                Eigen::FullPivHouseholderQRPreconditioner>
                decomp(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

            return decomp.solve(b);
        }

        /** Returns the amount of equations provided by this system.
         *  Only valid after A and b have been set appropriately.
         *  @return number of equations */
        size_t equations() const
        {
            return b.size();
        }

        /** Returns the amount of unknowns in this system.
         *  Only valid after A and b have been set appropriately.
         *  @return number of unknowns */
        size_t unknowns() const
        {
            return A.cols();
        }

        /** Checks if the system is underdetermined, i.e. equations < unknowns.
         *  Only valid after A and b have been set appropriately.
         *  @return true if the system is undertermined, else false */
        bool underdetermined() const
        {
            return unknowns() - equations() > 0;
        }
    };
}

#endif
