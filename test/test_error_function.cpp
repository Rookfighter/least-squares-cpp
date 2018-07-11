/*
 * test_error_function.cpp
 *
 *  Created on: 22 Jun 2018
 *      Author: Fabian Meyer
 */

#include "eigen_assert.h"
#include "error_functions.h"

using namespace opt;

TEST_CASE("Error functions")
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

        SECTION("evaluate functions")
        {
            Eigen::VectorXd state(3);
            state << 3, 2, 1;

            Eigen::VectorXd valExp(3);
            valExp << 8, -4, 8;

            Eigen::MatrixXd jacExp(3, 3);
            jacExp << 3, 0, -1, 0, -3, 2, 4, -2, 0;

            auto errRes = evalErrorFuncs(state, errFuncs);

            REQUIRE_MAT(valExp, errRes.val, eps);
            REQUIRE_MAT(jacExp, errRes.jac, eps);
        }

        SECTION("calculate squared error")
        {
            Eigen::VectorXd state(3);
            state << 3, 2, 1;
            double errExp = 72.0;

            auto errRes = evalErrorFuncs(state, errFuncs);
            double err = squaredError(errRes.val);

            REQUIRE(Approx(errExp).margin(eps) == err);
        }
    }
}
