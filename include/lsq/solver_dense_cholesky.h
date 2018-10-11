/*
 * solver_dense_cholesky.h
 *
 *  Created on: 11 Oct 2018
 *      Author: Fabian Meyer
 */

#ifndef LSQ_SOLVER_DENSE_CHOLESKY_H_
#define LSQ_SOLVER_DENSE_CHOLESKY_H_

#include <Eigen/Cholesky>
#include "lsq/solver.h"

namespace lsq
{
    class SolverDenseCholesky : public Solver
    {
    public:
        void solve(const LinearEquationSystem &system,
            Eigen::VectorXd &result) const override
        {
            Eigen::LDLT<Eigen::MatrixXd> decomp;
            decomp.compute(system.A);

            if(!decomp.isPositive())
                throw std::runtime_error("SolverDenseCholesky: matrix is not positive semi-definite");

            result = decomp.solve(system.b);
        }
    };
}

#endif
