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
    template<typename Scalar>
    class SolverDenseCholesky : public Solver<Scalar>
    {
    public:
        void solve(const LinearEquationSystem<Scalar> &system,
            Vector<Scalar> &result) const override
        {
            Eigen::LDLT<Matrix<Scalar>> decomp;
            decomp.compute(system.A);

            if(!decomp.isPositive())
                throw std::runtime_error("SolverDenseCholesky: matrix is not positive semi-definite");

            result = decomp.solve(system.b);
        }
    };
}

#endif
