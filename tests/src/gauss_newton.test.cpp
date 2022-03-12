/// gauss_newton.test.cpp
///
/// Author:     Fabian Meyer
/// Created On: 05 Aug 2019
/// License:    MIT


#include <lsqcpp/lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsqcpp;

TEMPLATE_TEST_CASE("gauss newton", "[algorithm]", float, double)
{
    constexpr auto Inputs = Eigen::Dynamic;
    using Scalar = TestType;
    constexpr auto eps = static_cast<Scalar>(1e-3);
    using Vector = Eigen::Matrix<Scalar, Inputs, 1>;

    SECTION("with jacobian")
    {
        SECTION("constant step size converge")
        {
            GaussNewtonX<Scalar, ParabolicError, ConstantStepFactor> optimizer;

            optimizer.setRefinementParameters({static_cast<Scalar>(0.5)});
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

        SECTION("armijo backtracking")
        {
            GaussNewtonX<Scalar, ParabolicError, ArmijoBacktracking> optimizer;

            optimizer.setRefinementParameters({static_cast<Scalar>(0.8),
                                               static_cast<Scalar>(1e-4),
                                               static_cast<Scalar>(1e-10),
                                               static_cast<Scalar>(1.0),
                                               50});
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

        SECTION("wolfe backtracking")
        {
            GaussNewtonX<Scalar, ParabolicError, WolfeBacktracking> optimizer;

            optimizer.setRefinementParameters({static_cast<Scalar>(0.8),
                                               static_cast<Scalar>(1e-4),
                                               static_cast<Scalar>(0.1),
                                               static_cast<Scalar>(1e-10),
                                               static_cast<Scalar>(1.0),
                                               100});
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
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
            REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE(Approx(errorExp).margin(eps) == result.error);
            REQUIRE(Approx(errorExp).margin(eps) == result.error);
        }

        SECTION("dogleg method")
        {
            // GaussNewtonX<Scalar, ParabolicError, DoglegMethod> optimizer;

            // optimizer.setRefinementParameters({static_cast<Scalar>(1),
            //                           static_cast<Scalar>(2),
            //                           static_cast<Scalar>(1e-9),
            //                           static_cast<Scalar>(0.25),
            //                           100});
            // optimizer.setMinimumStepLength(static_cast<Scalar>(1e-4));
            // optimizer.setMinimumGradientLength(static_cast<Scalar>(1e-4));
            // optimizer.setMaximumIterations(100);

            // Vector initGuess(4);
            // initGuess << 2, 1, 3, 4;

            // Scalar errorExp = 0;
            // Vector fvalExp = Vector::Zero(2);
            // Vector xvalExp = Vector::Zero(4);

            // auto result = optimizer.minimize(initGuess);

            // REQUIRE(result.converged);
            // REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
            // REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
            // REQUIRE(Approx(errorExp).margin(eps) == result.error);
            // REQUIRE(Approx(errorExp).margin(eps) == result.error);
        }
    }

    SECTION("without jacobian")
    {
        SECTION("constant step size converge")
        {
            GaussNewtonX<Scalar, ParabolicErrorNoJacobian, ConstantStepFactor> optimizer;

            optimizer.setRefinementParameters({static_cast<Scalar>(0.5)});
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
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1e-1);
            REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE(Approx(errorExp).margin(eps) == result.error);
            REQUIRE(Approx(errorExp).margin(eps) == result.error);
        }

        SECTION("armijo backtracking")
        {
            GaussNewtonX<Scalar, ParabolicErrorNoJacobian, ArmijoBacktracking> optimizer;

            optimizer.setRefinementParameters({static_cast<Scalar>(0.8),
                                               static_cast<Scalar>(1e-4),
                                               static_cast<Scalar>(1e-10),
                                               static_cast<Scalar>(1.0),
                                               50});
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
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1e-1);
            REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE(Approx(errorExp).margin(eps) == result.error);
            REQUIRE(Approx(errorExp).margin(eps) == result.error);
        }

        SECTION("wolfe backtracking")
        {
            GaussNewtonX<Scalar, ParabolicErrorNoJacobian, WolfeBacktracking> optimizer;

            optimizer.setRefinementParameters({static_cast<Scalar>(0.8),
                                               static_cast<Scalar>(1e-4),
                                               static_cast<Scalar>(0.1),
                                               static_cast<Scalar>(1e-10),
                                               static_cast<Scalar>(1.0),
                                               100});
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
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
            REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE(Approx(errorExp).margin(eps) == result.error);
            REQUIRE(Approx(errorExp).margin(eps) == result.error);
        }
    }

    SECTION("inverse jacobian")
    {
        SECTION("constant step size converge")
        {
            GaussNewtonX<Scalar, ParabolicErrorInverseJacobian, ConstantStepFactor> optimizer;

            optimizer.setRefinementParameters({static_cast<Scalar>(0.5)});
            optimizer.setMinimumStepLength(static_cast<Scalar>(1e-10));
            optimizer.setMinimumGradientLength(static_cast<Scalar>(1e-10));
            optimizer.setMaximumIterations(100);

            Vector initGuess(4);
            initGuess << 2, 1, 3, 4;

            Scalar errorExp = 0;
            Vector fvalExp = Vector::Zero(2);
            Vector xvalExp = Vector::Zero(4);

            auto result = optimizer.minimize(initGuess);

            REQUIRE_FALSE(result.converged);
            REQUIRE_NOT_MATRIX_APPROX(xvalExp, result.xval, eps);
            REQUIRE_NOT_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE(Approx(errorExp).margin(eps) != result.error);
            REQUIRE(Approx(errorExp).margin(eps) != result.error);
        }

        SECTION("armijo backtracking")
        {
            GaussNewtonX<Scalar, ParabolicErrorInverseJacobian, ArmijoBacktracking> optimizer;

            optimizer.setRefinementParameters({static_cast<Scalar>(0.8),
                                               static_cast<Scalar>(1e-4),
                                               static_cast<Scalar>(1e-10),
                                               static_cast<Scalar>(1.0),
                                               50});
            optimizer.setMinimumStepLength(static_cast<Scalar>(1e-10));
            optimizer.setMinimumGradientLength(static_cast<Scalar>(1e-10));
            optimizer.setMaximumIterations(100);

            Vector initGuess(4);
            initGuess << 2, 1, 3, 4;

            Scalar errorExp = 0;
            Vector fvalExp = Vector::Zero(2);
            Vector xvalExp = Vector::Zero(4);

            auto result = optimizer.minimize(initGuess);

            REQUIRE_FALSE(result.converged);
            REQUIRE_NOT_MATRIX_APPROX(xvalExp, result.xval, eps);
            REQUIRE_NOT_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE(Approx(errorExp).margin(eps) != result.error);
            REQUIRE(Approx(errorExp).margin(eps) != result.error);
        }

        SECTION("wolfe backtracking")
        {
            GaussNewtonX<Scalar, ParabolicErrorInverseJacobian, WolfeBacktracking> optimizer;

            optimizer.setRefinementParameters({static_cast<Scalar>(0.8),
                                               static_cast<Scalar>(1e-4),
                                               static_cast<Scalar>(0.1),
                                               static_cast<Scalar>(1e-10),
                                               static_cast<Scalar>(1.0),
                                               100});
            optimizer.setMinimumStepLength(static_cast<Scalar>(1e-10));
            optimizer.setMinimumGradientLength(static_cast<Scalar>(1e-10));
            optimizer.setMaximumIterations(100);

            Vector initGuess(4);
            initGuess << 2, 1, 3, 4;

            Scalar errorExp = 0;
            Vector fvalExp = Vector::Zero(2);
            Vector xvalExp = Vector::Zero(4);

            auto result = optimizer.minimize(initGuess);

            REQUIRE_FALSE(result.converged);
            REQUIRE_NOT_MATRIX_APPROX(xvalExp, result.xval, eps);
            REQUIRE_NOT_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE(Approx(errorExp).margin(eps) != result.error);
            REQUIRE(Approx(errorExp).margin(eps) != result.error);
        }
    }
}
