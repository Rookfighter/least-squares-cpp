/*
 * solver_dense_svd.h
 *
 *  Created on: 11 Oct 2018
 *      Author: Fabian Meyer
 */

#ifndef LSQ_SOLVER_DENSE_SVD_H_
#define LSQ_SOLVER_DENSE_SVD_H_

#include "lsq/solver.h"

namespace lsq
{
    class SolverDenseSVD : public Solver
    {
    public:
        void solve(const LinearEquationSystem &system,
            Eigen::VectorXd &result) const override
        {
            Eigen::JacobiSVD<Eigen::MatrixXd,
                Eigen::FullPivHouseholderQRPreconditioner>
                decomp(system.A, Eigen::ComputeFullU | Eigen::ComputeFullV);

            result = decomp.solve(system.b);
        }
    };
}

#endif
