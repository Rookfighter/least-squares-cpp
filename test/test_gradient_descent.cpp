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
    SECTION("with linear error functions")
    {
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
        GradientDescent gd;
        gd.setDamping(1.0);
        gd.setLineSearchAlgorithm(new IncreasingLineSearch());
        gd.setErrorFunctions(errFuncs);
        gd.setMaxIterations(20);
        gd.setEpsilon(1e-6);

        SECTION("optimize")
        {
            Eigen::VectorXd errValA, errValB;
            Eigen::MatrixXd errJacA, errJacB;
            Eigen::VectorXd state(3);
            state << 3, 2, 1;
            Eigen::VectorXd stateExp(3);
            stateExp << 1, 2, 3;

            auto result = gd.optimize(state);

            evalErrorFuncs(state, errFuncs, errValA, errJacA);
            evalErrorFuncs(result.state, errFuncs, errValB, errJacB);

            REQUIRE(result.converged);
            REQUIRE(result.iterations == 10);
            // gradient method shows decrease in error
            REQUIRE(squaredError(errValB) < squaredError(errValA));
        }
    }

    SECTION("with linear error functions without jacobians")
    {
        LinearErrFuncNoJac *eq1 = new LinearErrFuncNoJac();
        LinearErrFuncNoJac *eq2 = new LinearErrFuncNoJac();
        LinearErrFuncNoJac *eq3 = new LinearErrFuncNoJac();

        eq1->factors.resize(3);
        eq1->factors << 3, 0, -1;

        eq2->factors.resize(3);
        eq2->factors << 0, -3, 2;

        eq3->factors.resize(3);
        eq3->factors << 4, -2, 0;

        std::vector<ErrorFunction *> errFuncs = {eq1, eq2, eq3};
        GradientDescent gd;
        gd.setDamping(1.0);
        gd.setLineSearchAlgorithm(new IncreasingLineSearch());
        gd.setErrorFunctions(errFuncs);
        gd.setMaxIterations(20);
        gd.setEpsilon(1e-6);

        SECTION("optimize")
        {
            Eigen::VectorXd errValA, errValB;
            Eigen::MatrixXd errJacA, errJacB;
            Eigen::VectorXd state(3);
            state << 3, 2, 1;
            Eigen::VectorXd stateExp(3);
            stateExp << 1, 2, 3;

            auto result = gd.optimize(state);

            evalErrorFuncs(state, errFuncs, errValA, errJacA);
            evalErrorFuncs(result.state, errFuncs, errValB, errJacB);

            REQUIRE(result.converged);
            REQUIRE(result.iterations == 10);
            // gradient method shows decrease in error
            REQUIRE(squaredError(errValB) < squaredError(errValA));
        }
    }
}
