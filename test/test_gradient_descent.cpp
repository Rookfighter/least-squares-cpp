/*
 * test_gradient_descent.cpp
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#include "eigen_assert.h"
#include "error_functions.h"
#include <lsq/gradient_descent.h>
#include <lsq/increasing_line_search.h>
#include <lsq/linear_equation_system.h>

using namespace lsq;

TEST_CASE("Gradient Descent")
{
    SECTION("with linear error function")
    {
        LinearErrFuncd *errFunc = new LinearErrFuncd();

        errFunc->factors.resize(3, 3);
        errFunc->factors << 3, 0, -1,
            0, -3, 2,
            4, -2, 0;
        errFunc->factors.transposeInPlace();

        GradientDescent<double> gd;
        gd.setDamping(1.0);
        gd.setLineSearchAlgorithm(new IncreasingLineSearch<double>());
        gd.setErrorFunction(errFunc);
        gd.setMaxIterations(20);
        gd.setEpsilon(1e-6);

        SECTION("optimize")
        {
            lsq::Vectord errValA, errValB;
            lsq::Matrixd errJacA, errJacB;
            lsq::Vectord state(3);
            state << 3, 2, 1;
            lsq::Vectord stateExp(3);
            stateExp << 1, 2, 3;

            auto result = gd.optimize(state);

            errFunc->evaluate(state, errValA, errJacA);
            errFunc->evaluate(result.state, errValB, errJacB);

            REQUIRE(result.converged);
            REQUIRE(result.iterations == 10);
            // gradient method shows decrease in error
            REQUIRE(squaredError<double>(errValB) < squaredError<double>(errValA));
        }
    }

    SECTION("with linear error functions without jacobians")
    {
        LinearErrFuncNoJacd *errFunc = new LinearErrFuncNoJacd();

        errFunc->factors.resize(3, 3);
        errFunc->factors << 3, 0, -1,
            0, -3, 2,
            4, -2, 0;
        errFunc->factors.transposeInPlace();

        GradientDescent<double> gd;
        gd.setDamping(1.0);
        gd.setLineSearchAlgorithm(new IncreasingLineSearch<double>());
        gd.setErrorFunction(errFunc);
        gd.setMaxIterations(20);
        gd.setEpsilon(1e-6);

        SECTION("optimize")
        {
            lsq::Vectord errValA, errValB;
            lsq::Matrixd errJacA, errJacB;
            lsq::Vectord state(3);
            state << 3, 2, 1;
            lsq::Vectord stateExp(3);
            stateExp << 1, 2, 3;

            auto result = gd.optimize(state);

            errFunc->evaluate(state, errValA, errJacA);
            errFunc->evaluate(result.state, errValB, errJacB);

            REQUIRE(result.converged);
            REQUIRE(result.iterations == 10);
            // gradient method shows decrease in error
            REQUIRE(squaredError<double>(errValB) < squaredError<double>(errValA));
        }
    }
}
