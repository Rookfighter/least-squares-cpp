/*
 * test_gradient_descent.cpp
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#include "eigen_assert.h"
#include "error_functions.h"
#include <optcpp/gradient_descent.h>
#include <optcpp/increasing_line_search.h>
#include <optcpp/linear_equation_system.h>

using namespace opt;

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

        SECTION("optimize")
        {
            Eigen::VectorXd state(3);
            state << 3, 2, 1;
            Eigen::VectorXd stateExp(3);
            stateExp << 1, 2, 3;

            auto result = gd.run(state, 1e-6, 10);

            auto errResA = evalErrorFuncs(state, errFuncs);
            auto errResB = evalErrorFuncs(result.state, errFuncs);

            // gradient descent does not converge
            REQUIRE(!result.converged);
            REQUIRE(result.iterations == 10);
            // gradient method shows decrease in error
            REQUIRE(squaredError(errResB.val) < squaredError(errResA.val));
        }
    }
}
