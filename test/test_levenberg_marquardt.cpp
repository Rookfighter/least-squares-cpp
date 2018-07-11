/*
 * test_gauss_newton.cpp
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#include "eigen_assert.h"
#include "error_functions.h"
#include <optcpp/levenberg_marquardt.h>

TEST_CASE("Levenberg Marquardt")
{
    SECTION("with linear error functions")
    {
        const double eps = 1e-6;

        LinearErrFunc *eq1 = new LinearErrFunc();
        LinearErrFunc *eq2 = new LinearErrFunc();
        LinearErrFunc *eq3 = new LinearErrFunc();

        eq1->factors.resize(3);
        eq1->factors << 3, 0, -1;

        eq2->factors.resize(3);
        eq2->factors << 0, -3, 2;

        eq3->factors.resize(3);
        eq3->factors << 4, -2, 0;

        std::vector<opt::ErrorFunction *> errFuncs = {eq1, eq2, eq3};
        opt::LevenbergMarquardt lm;
        lm.setDamping(1.0);
        lm.setErrorFunctions(errFuncs);

        SECTION("optimize")
        {
            Eigen::VectorXd state(3);
            state << 3, 2, 1;
            Eigen::VectorXd stateExp(3);
            stateExp << 1, 2, 3;

            auto result = lm.run(state, 1e-12, 10);
            state << result.state(0) / result.state(0),
                result.state(1) / result.state(0),
                result.state(2) / result.state(0);

            REQUIRE(result.converged);
            REQUIRE(result.iterations == 7);
            REQUIRE_MAT(stateExp, state, eps);
        }
    }
}
