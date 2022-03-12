/// levenberg_marquardt.test.cpp
///
/// Author:     Fabian Meyer
/// Created On: 04 Feb 2022
/// License:    MIT

#include <lsqcpp/lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsqcpp;

TEMPLATE_TEST_CASE("levenberg marquardt", "[algorithm]", float, double)
{
    using Scalar = TestType;
    using Parameter = LevenbergMarquardtParameter<Scalar>;

    SECTION("parameter")
    {
        SECTION("construction")
        {
            SECTION("default")
            {
                Parameter param;
                REQUIRE(param.initialLambda() == static_cast<Scalar>(1));
                REQUIRE(param.increase() == static_cast<Scalar>(2));
                REQUIRE(param.decrease() == static_cast<Scalar>(0.5));
                REQUIRE(param.maximumIterations() == 0);
            }

            SECTION("parametrized A")
            {
                Parameter param(static_cast<Scalar>(0.42),
                                static_cast<Scalar>(2.25),
                                static_cast<Scalar>(0.72),
                                10);
                REQUIRE(param.initialLambda() == static_cast<Scalar>(0.42));
                REQUIRE(param.increase() == static_cast<Scalar>(2.25));
                REQUIRE(param.decrease() == static_cast<Scalar>(0.72));
                REQUIRE(param.maximumIterations() == 10);
            }
        }

        SECTION("setter")
        {
            SECTION("lambda")
            {
                Parameter param;
                REQUIRE(param.initialLambda() == static_cast<Scalar>(1));

                param.setInitialLambda(static_cast<Scalar>(0.42));
                REQUIRE(param.initialLambda() == static_cast<Scalar>(0.42));
            }

            SECTION("increase")
            {
                Parameter param;
                REQUIRE(param.increase() == static_cast<Scalar>(2));

                param.setIncrease(static_cast<Scalar>(2.25));
                REQUIRE(param.increase() == static_cast<Scalar>(2.25));
            }

            SECTION("decrease")
            {
                Parameter param;
                REQUIRE(param.decrease() == static_cast<Scalar>(0.5));

                param.setDecrease(static_cast<Scalar>(0.72));
                REQUIRE(param.decrease() == static_cast<Scalar>(0.72));
            }

            SECTION("maximum iterations")
            {
                Parameter param;
                REQUIRE(param.maximumIterations() == 0);

                param.setMaximumIterations(10);
                REQUIRE(param.maximumIterations() == 10);
            }
        }
    }

    SECTION("refinement")
    {
        constexpr auto Inputs = Eigen::Dynamic;
        constexpr auto eps = static_cast<Scalar>(1e-3);
        using Vector = Eigen::Matrix<Scalar, Inputs, 1>;

        SECTION("with jacobian")
        {
            LevenbergMarquardtX<Scalar, ParabolicError> optimizer;

            optimizer.setMinimumStepLength(static_cast<Scalar>(1e-10));
            optimizer.setMinimumGradientLength(static_cast<Scalar>(1e-10));
            optimizer.setMaximumIterations(100);

            Vector initGuess(4);
            initGuess << 2, 1, 3, 4;

            Scalar errorExp = 0;
            Vector fvalExp = Vector::Zero(2);
            Vector xvalExp = Vector::Zero(4);

            auto result = optimizer.minimize(initGuess);

            REQUIRE(result.converged);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, static_cast<Scalar>(1e-1));
            REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE(Approx(errorExp).margin(eps) == result.error);
            REQUIRE(Approx(errorExp).margin(eps) == result.error);
        }

        SECTION("without jacobian")
        {
            LevenbergMarquardtX<Scalar, ParabolicErrorNoJacobian> optimizer;

            optimizer.setMinimumStepLength(static_cast<Scalar>(1e-10));
            optimizer.setMinimumGradientLength(static_cast<Scalar>(1e-10));
            optimizer.setMaximumIterations(100);

            Vector initGuess(4);
            initGuess << 2, 1, 3, 4;

            Scalar errorExp = 0;
            Vector fvalExp = Vector::Zero(2);
            Vector xvalExp = Vector::Zero(4);

            auto result = optimizer.minimize(initGuess);

            REQUIRE(result.converged);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, static_cast<Scalar>(1e-1));
            REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE(Approx(errorExp).margin(eps) == result.error);
            REQUIRE(Approx(errorExp).margin(eps) == result.error);
        }
    }

}