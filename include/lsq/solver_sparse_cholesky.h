/*
 * solver_dense_cholesky.h
 *
 *  Created on: 11 Oct 2018
 *      Author: Fabian Meyer
 */

#ifndef LSQ_SOLVER_DENSE_CHOLESKY_H_
#define LSQ_SOLVER_DENSE_CHOLESKY_H_

#include <Eigen/SparseCholesky>
#include "lsq/solver.h"

namespace lsq
{
    class SolverDenseCholesky : public Solver
    {
    private:
        typedef Eigen::SparseMatrix<double> SparseMatrix;
    public:
        void solve(const LinearEquationSystem &system,
            Eigen::VectorXd &result) const override
        {
            Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper> decomp;

            SparseMatrix sparseA = system.A.sparseView();

            decomp.compute(sparseA);

            if(decomp.info() != Eigen::Success)
                throw std::runtime_error("SolverSparseCholesky: cholesky decomposition failed");

            result = decomp.solve(system.b);
        }
    };
}

#endif
