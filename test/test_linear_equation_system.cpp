/*
 * test_linear_equation_system.cpp
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#include <optcpp/linear_equation_system.h>
#include "error_functions.h"
#include "eigen_assert.h"

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

        std::vector<opt::ErrorFunction*> errFuncs = {&eq1, &eq2, &eq3};

        SECTION("construct function")
        {
            Eigen::VectorXd state(3);
            state << 3, 2, 1;

            Eigen::VectorXd bexp(3);
            bexp << 8, -4, 8;

            Eigen::MatrixXd Aexp(3,3);
            Aexp << 3, 0, -1,
                    0, -3, 2,
                    4, -2, 0;

            opt::LinearEquationSystem eqSys;
            eqSys.construct(state, errFuncs);

            REQUIRE(3 == eqSys.unknowns());
            REQUIRE(3 == eqSys.equations());
            REQUIRE(!eqSys.underdetermined());

            REQUIRE_MAT(bexp, eqSys.b, eps);
            REQUIRE_MAT(Aexp, eqSys.A, eps);
        }

        SECTION("construct constructor")
        {
            Eigen::VectorXd state(3);
            state << 3, 2, 1;

            Eigen::VectorXd bexp(3);
            bexp << 8, -4, 8;

            Eigen::MatrixXd Aexp(3,3);
            Aexp << 3, 0, -1,
                    0, -3, 2,
                    4, -2, 0;

            opt::LinearEquationSystem eqSys(state, errFuncs);

            REQUIRE(3 == eqSys.unknowns());
            REQUIRE(3 == eqSys.equations());
            REQUIRE(!eqSys.underdetermined());

            REQUIRE_MAT(bexp, eqSys.b, eps);
            REQUIRE_MAT(Aexp, eqSys.A, eps);
        }

        SECTION("solve non underdetermined")
        {
            Eigen::VectorXd state(3);
            state << 3, 2, 1;

            Eigen::VectorXd errExp = Eigen::VectorXd::Zero(3);

            opt::LinearEquationSystem eqSys(state, errFuncs);
            REQUIRE(!eqSys.underdetermined());

            Eigen::VectorXd step = eqSys.solveSVD();
            state -= step;

            eqSys.construct(state, errFuncs);

            REQUIRE_MAT(errExp, eqSys.b, eps);
        }
    }
}
