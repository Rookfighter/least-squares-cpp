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

        LinearErrFuncd errFunc;
        errFunc.factors.resize(3, 3);
        errFunc.factors << 3, 0, -1,
            0, -3, 2,
            4, -2, 0;
        errFunc.factors.transposeInPlace();

        SECTION("evaluate functions")
        {
            lsq::Vectord errVal;
            lsq::Matrixd errJac;

            lsq::Vectord state(3);
            state << 3, 2, 1;

            lsq::Vectord valExp(3);
            valExp << 8, -4, 8;

            lsq::Matrixd jacExp(3, 3);
            jacExp << 3, 0, -1, 0, -3, 2, 4, -2, 0;

            errFunc.evaluate(state, errVal, errJac);

            REQUIRE_MAT(valExp, errVal, eps);
            REQUIRE_MAT(jacExp, errJac, eps);
        }

        SECTION("calculate squared error")
        {
            lsq::Vectord errVal;
            lsq::Matrixd errJac;
            lsq::Vectord state(3);
            state << 3, 2, 1;
            double errExp = 72.0;

            errFunc.evaluate(state, errVal, errJac);
            double err = squaredError<double>(errVal);

            REQUIRE(Approx(errExp).margin(eps) == err);
        }

        SECTION("finite differences")
        {
            lsq::Vectord errVal;
            lsq::Matrixd expJac;
            lsq::Matrixd actJac;
            lsq::Vectord state(3);
            state << 3, 2, 1;

            errFunc.setNumericalEps(1e-8);
            errFunc.evaluate(state, errVal, expJac);
            errFunc.computeFiniteDifferences(state, errVal, actJac);

            REQUIRE_MAT(expJac, actJac, eps);
        }
    }

    SECTION("linear without jacobians")
    {
        const double eps = 1e-6;

        LinearErrFuncNoJacd errFunc;

        errFunc.factors.resize(3, 3);
        errFunc.factors << 3, 0, -1,
            0, -3, 2,
            4, -2, 0;
        errFunc.factors.transposeInPlace();

        SECTION("evaluate functions")
        {
            lsq::Vectord errVal;
            lsq::Matrixd errJac;

            lsq::Vectord state(3);
            state << 3, 2, 1;

            lsq::Vectord valExp(3);
            valExp << 8, -4, 8;

            lsq::Matrixd jacExp(3, 3);
            jacExp << 3, 0, -1, 0, -3, 2, 4, -2, 0;

            errFunc.evaluate(state, errVal, errJac);

            REQUIRE_MAT(valExp, errVal, eps);
            REQUIRE_MAT(jacExp, errJac, eps);
        }

        SECTION("calculate squared error")
        {
            lsq::Vectord errVal;
            lsq::Matrixd errJac;
            lsq::Vectord state(3);
            state << 3, 2, 1;
            double errExp = 72.0;

            errFunc.evaluate(state, errVal, errJac);
            double err = squaredError(errVal);

            REQUIRE(Approx(errExp).margin(eps) == err);
        }
    }
}
