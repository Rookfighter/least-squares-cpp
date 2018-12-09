/*
 * test_gauss_newton.cpp
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#include "eigen_assert.h"
#include "error_functions.h"
#include <lsq/levenberg_marquardt.h>

using namespace lsq;

TEST_CASE("Levenberg Marquardt")
{
    SECTION("with linear error function")
    {
        LinearErrFuncd *errFunc = new LinearErrFuncd();

        errFunc->factors.resize(3, 3);
        errFunc->factors << 3, 0, -1,
            0, -3, 2,
            4, -2, 0;
        errFunc->factors.transposeInPlace();

        LevenbergMarquardt<double> lm;
        lm.setDamping(1.0);
        lm.setErrorFunction(errFunc);
        lm.setMaxIterations(10);
        lm.setEpsilon(1e-12);

        SECTION("optimize")
        {
            lsq::Vectord errValA, errValB;
            lsq::Matrixd errJacA, errJacB;
            lsq::Vectord state(3);
            state << 3, 2, 1;
            lsq::Vectord stateExp(3);
            stateExp << 1, 2, 3;

            auto result = lm.optimize(state);

            errFunc->evaluate(state, errValA, errJacA);
            errFunc->evaluate(result.state, errValB, errJacB);

            REQUIRE(result.converged);
            REQUIRE(result.iterations == 7);
            REQUIRE(squaredError(errValB) < squaredError<double>(errValA));
        }
    }
}
