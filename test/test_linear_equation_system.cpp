/*
 * test_linear_equation_system.cpp
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#include "eigen_assert.h"
#include "error_functions.h"
#include <optcpp/linear_equation_system.h>

using namespace opt;

TEST_CASE("Linear equation system")
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
            Eigen::VectorXd state(3);
            state << 3, 2, 1;

            Eigen::VectorXd errExp = Eigen::VectorXd::Zero(3);

            auto errRes = evalErrorFuncs(state, errFuncs);
            LinearEquationSystem eqSys;
            eqSys.b = errRes.jac.transpose() * errRes.val;
            eqSys.A = errRes.jac.transpose() * errRes.jac;

            REQUIRE(!eqSys.underdetermined());

            Eigen::VectorXd step = eqSys.solveSVD();
            state -= step;

            errRes = evalErrorFuncs(state, errFuncs);

            REQUIRE_MAT(errExp, errRes.val, eps);
        }
    }
}
