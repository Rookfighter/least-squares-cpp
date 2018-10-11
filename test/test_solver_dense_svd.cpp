/*
 * test_linear_equation_system.cpp
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#include "eigen_assert.h"
#include "error_functions.h"
#include <lsq/solver_dense_svd.h>

using namespace lsq;

TEST_CASE("Solver Dense SVD")
{
    SECTION("with linear error functions")
    {
        const double eps = 1e-6;

        LinearErrFunc eq1;
        LinearErrFunc eq2;
        LinearErrFunc eq3;

        eq1.factors.resize(3);
        eq1.factors << 3, 0, -1;

        eq2.factors.resize(3);
        eq2.factors << 0, -3, 2;

        eq3.factors.resize(3);
        eq3.factors << 4, -2, 0;

        std::vector<ErrorFunction *> errFuncs = {&eq1, &eq2, &eq3};

        SECTION("solve non underdetermined")
        {
            Eigen::VectorXd errVal;
            Eigen::MatrixXd errJac;
            Eigen::VectorXd state(3);
            state << 3, 2, 1;

            Eigen::VectorXd errExp = Eigen::VectorXd::Zero(3);

            evalErrorFuncs(state, errFuncs, errVal, errJac);
            LinearEquationSystem eqSys;
            eqSys.b = errJac.transpose() * errVal;
            eqSys.A = errJac.transpose() * errJac;

            REQUIRE(!eqSys.underdetermined());

            SolverDenseSVD solver;
            Eigen::VectorXd step ;
            solver.solve(eqSys, step);
            state -= step;

            evalErrorFuncs(state, errFuncs, errVal, errJac);

            REQUIRE_MAT(errExp, errVal, eps);
        }
    }
}
