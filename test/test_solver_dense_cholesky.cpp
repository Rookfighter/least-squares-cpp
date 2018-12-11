/*
 * test_solver_dense_cholesky.cpp
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#include "eigen_assert.h"
#include "error_functions.h"
#include <lsq/solver_dense_cholesky.h>

using namespace lsq;

TEST_CASE("Solver Dense Cholesky")
{
    SECTION("with linear error function")
    {
        const double eps = 1e-6;

        LinearErrFuncd errFunc;

        errFunc.factors.resize(3, 3);
        errFunc.factors << 3, 0, -1,
            0, -3, 2,
            4, -2, 0;

        SECTION("solve non underdetermined")
        {
            lsq::Vectord errVal;
            lsq::Matrixd errJac;
            lsq::Vectord state(3);
            state << 3, 2, 1;

            lsq::Vectord errExp = lsq::Vectord::Zero(3);

            errFunc.evaluate(state, errVal, errJac);
            LinearEquationSystem<double> eqSys;
            eqSys.b = errJac.transpose() * errVal;
            eqSys.A = errJac.transpose() * errJac;

            REQUIRE(!eqSys.underdetermined());

            SolverDenseCholesky<double> solver;
            lsq::Vectord step ;
            solver.solve(eqSys, step);
            state -= step;

            errFunc.evaluate(state, errVal, errJac);

            REQUIRE_MAT(errExp, errVal, eps);
        }
    }
}
