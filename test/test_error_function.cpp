/*
 * test_error_function.cpp
 *
 *  Created on: 22 Jun 2018
 *      Author: Fabian Meyer
 */

#include "eigen_assert.h"
#include "error_functions.h"

using namespace lsq;

TEST_CASE("Error functions")
{
    SECTION("linear with jacobians")
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
            Eigen::VectorXd errVal;
            Eigen::MatrixXd errJac;

            Eigen::VectorXd state(3);
            state << 3, 2, 1;

            Eigen::VectorXd valExp(3);
            valExp << 8, -4, 8;

            Eigen::MatrixXd jacExp(3, 3);
            jacExp << 3, 0, -1, 0, -3, 2, 4, -2, 0;

            evalErrorFuncs(state, errFuncs, errVal, errJac);

            REQUIRE_MAT(valExp, errVal, eps);
            REQUIRE_MAT(jacExp, errJac, eps);
        }

        SECTION("calculate squared error")
        {
            Eigen::VectorXd errVal;
            Eigen::MatrixXd errJac;
            Eigen::VectorXd state(3);
            state << 3, 2, 1;
            double errExp = 72.0;

            evalErrorFuncs(state, errFuncs, errVal, errJac);
            double err = squaredError(errVal);

            REQUIRE(Approx(errExp).margin(eps) == err);
        }

        SECTION("finite differences")
        {
            Eigen::VectorXd errVal;
            Eigen::MatrixXd expJac;
            Eigen::MatrixXd actJac;
            Eigen::VectorXd state(3);
            state << 3, 2, 1;

            eq1._evaluate(state, errVal, expJac);
            eq1.computeFiniteDifferences(state, errVal, actJac, 1e-8);

            REQUIRE_MAT(expJac, actJac, eps);

            eq2._evaluate(state, errVal, expJac);
            eq2.computeFiniteDifferences(state, errVal, actJac, 1e-8);

            REQUIRE_MAT(expJac, actJac, eps);

            eq3._evaluate(state, errVal, expJac);
            eq3.computeFiniteDifferences(state, errVal, actJac, 1e-8);

            REQUIRE_MAT(expJac, actJac, eps);
        }
    }

    SECTION("linear without jacobians")
    {
        const double eps = 1e-6;

        LinearErrFuncNoJac eq1;
        LinearErrFuncNoJac eq2;
        LinearErrFuncNoJac eq3;

        eq1.factors.resize(3);
        eq1.factors << 3, 0, -1;

        eq2.factors.resize(3);
        eq2.factors << 0, -3, 2;

        eq3.factors.resize(3);
        eq3.factors << 4, -2, 0;

        std::vector<ErrorFunction *> errFuncs = {&eq1, &eq2, &eq3};

        SECTION("evaluate functions")
        {
            Eigen::VectorXd errVal;
            Eigen::MatrixXd errJac;

            Eigen::VectorXd state(3);
            state << 3, 2, 1;

            Eigen::VectorXd valExp(3);
            valExp << 8, -4, 8;

            Eigen::MatrixXd jacExp(3, 3);
            jacExp << 3, 0, -1, 0, -3, 2, 4, -2, 0;

            evalErrorFuncs(state, errFuncs, errVal, errJac);

            REQUIRE_MAT(valExp, errVal, eps);
            REQUIRE_MAT(jacExp, errJac, eps);
        }

        SECTION("calculate squared error")
        {
            Eigen::VectorXd errVal;
            Eigen::MatrixXd errJac;
            Eigen::VectorXd state(3);
            state << 3, 2, 1;
            double errExp = 72.0;

            evalErrorFuncs(state, errFuncs, errVal, errJac);
            double err = squaredError(errVal);

            REQUIRE(Approx(errExp).margin(eps) == err);
        }
    }
}
