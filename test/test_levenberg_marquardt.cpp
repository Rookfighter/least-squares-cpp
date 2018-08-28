/*
 * test_gauss_newton.cpp
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#include "eigen_assert.h"
#include "error_functions.h"
#include <optcpp/levenberg_marquardt.h>

using namespace opt;

TEST_CASE("Levenberg Marquardt")
{
    SECTION("with linear error functions")
    {
        // const double eps = 1e-6;

        LinearErrFunc *eq1 = new LinearErrFunc();
        LinearErrFunc *eq2 = new LinearErrFunc();
        LinearErrFunc *eq3 = new LinearErrFunc();

        eq1->factors.resize(3);
        eq1->factors << 3, 0, -1;

        eq2->factors.resize(3);
        eq2->factors << 0, -3, 2;

        eq3->factors.resize(3);
        eq3->factors << 4, -2, 0;

        std::vector<ErrorFunction *> errFuncs = {eq1, eq2, eq3};
        LevenbergMarquardt lm;
        lm.setDamping(1.0);
        lm.setErrorFunctions(errFuncs);
        lm.setMaxIterations(10);
        lm.setEpsilon(1e-12);

        SECTION("optimize")
        {
            Eigen::VectorXd errValA, errValB;
            Eigen::MatrixXd errJacA, errJacB;
            Eigen::VectorXd state(3);
            state << 3, 2, 1;
            Eigen::VectorXd stateExp(3);
            stateExp << 1, 2, 3;

            auto result = lm.optimize(state);

            evalErrorFuncs(state, errFuncs, errValA, errJacA);
            evalErrorFuncs(result.state, errFuncs, errValB, errJacB);

            REQUIRE(result.converged);
            REQUIRE(result.iterations == 7);
            REQUIRE(squaredError(errValB) < squaredError(errValA));
        }
    }
}
