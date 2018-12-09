/*
 * test_gauss_newton.cpp
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#include "eigen_assert.h"
#include "error_functions.h"
#include <lsq/gauss_newton.h>

TEST_CASE("Gauss Newton")
{
    SECTION("with linear error function")
    {
        const double eps = 1e-6;

        LinearErrFuncd *errFunc = new LinearErrFuncd();

        errFunc->factors.resize(3, 3);
        errFunc->factors << 3, 0, -1,
            0, -3, 2,
            4, -2, 0;
        errFunc->factors.transposeInPlace();

        lsq::GaussNewton<double> gn;
        gn.setDamping(1.0);
        gn.setErrorFunction(errFunc);
        gn.setMaxIterations(10);
        gn.setEpsilon(1e-12);

        SECTION("optimize")
        {
            lsq::Vectord state(3);
            state << 3, 2, 1;
            lsq::Vectord stateExp(3);
            stateExp << 1, 2, 3;

            auto result = gn.optimize(state);
            state << result.state(0) / result.state(0),
                result.state(1) / result.state(0),
                result.state(2) / result.state(0);

            REQUIRE(result.converged);
            REQUIRE(result.iterations == 1);
            REQUIRE_MAT(stateExp, state, eps);
        }
    }
}
