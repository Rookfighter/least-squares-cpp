/// optimization.cpp
///
/// Author: Fabian Meyer
/// Created On: 05 Aug 2019


#include <lsqcpp/lsqcpp.hpp>
#include "eigen_require.hpp"
#include "parabolic_error.hpp"

using namespace lsqcpp;

TEMPLATE_TEST_CASE("gradient descent", "[algorithm]", float, double)
{
    using Scalar = TestType;
    const auto eps = static_cast<Scalar>(1e-3);
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    SECTION("with jacobian")
    {
        SECTION("constant step size")
        {
            GradientDescentX<Scalar, ParabolicError, ConstantStepFactor> optimizer;

            optimizer.setRefinementParameters({static_cast<Scalar>(1e-2)});
            optimizer.setMinimumStepLength(static_cast<Scalar>(1e-10));
            optimizer.setMinimumGradientLength(static_cast<Scalar>(1e-10));
            optimizer.setMaximumIterations(100);

            Vector initGuess(4);
            initGuess << 2, 1, 3, 4;

            Vector fvalExp = Vector::Zero(2);
            Vector xvalExp = Vector::Zero(4);

            auto result = optimizer.minimize(initGuess);

            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1);
            REQUIRE_MATRIX_APPROX(fvalExp, result.fval, 1);
            REQUIRE(Approx(Scalar{0}).margin(0.1) == result.error);
        }

        SECTION("barzilai borwein direct")
        {
            GradientDescentX<Scalar, ParabolicError, BarzilaiBorwein> optimizer;

            optimizer.setRefinementParameters({BarzilaiBorwein::Mode::Direct, static_cast<Scalar>(1e-2)});
            optimizer.setMinimumStepLength(static_cast<Scalar>(1e-10));
            optimizer.setMinimumGradientLength(static_cast<Scalar>(1e-10));
            optimizer.setMaximumIterations(100);

            Vector initGuess(4);
            initGuess << 2, 1, 3, 4;

            Vector fvalExp = Vector::Zero(2);
            Vector xvalExp = Vector::Zero(4);

            auto result = optimizer.minimize(initGuess);

            REQUIRE(result.converged);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
            REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE(Approx(Scalar{0}).margin(eps) == result.error);
        }

        SECTION("barzilai borwein inverse")
        {
            GradientDescentX<Scalar, ParabolicError, BarzilaiBorwein> optimizer;

            optimizer.setRefinementParameters({BarzilaiBorwein::Mode::Inverse, static_cast<Scalar>(1e-2)});
            optimizer.setMinimumStepLength(static_cast<Scalar>(1e-10));
            optimizer.setMinimumGradientLength(static_cast<Scalar>(1e-10));
            optimizer.setMaximumIterations(100);

            Vector initGuess(4);
            initGuess << 2, 1, 3, 4;

            Vector fvalExp = Vector::Zero(2);
            Vector xvalExp = Vector::Zero(4);

            auto result = optimizer.minimize(initGuess);

            REQUIRE(result.converged);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
            REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE(Approx(Scalar{0}).margin(eps) == result.error);
        }
    }

    SECTION("without jacobian")
    {
        SECTION("constant step size")
        {
            GradientDescentX<Scalar, ParabolicErrorNoJacobian, ConstantStepFactor> optimizer;

            optimizer.setRefinementParameters({static_cast<Scalar>(1e-2)});
            optimizer.setMinimumStepLength(static_cast<Scalar>(1e-10));
            optimizer.setMinimumGradientLength(static_cast<Scalar>(1e-10));
            optimizer.setMaximumIterations(100);

            Vector initGuess(4);
            initGuess << 2, 1, 3, 4;

            Vector fvalExp = Vector::Zero(2);
            Vector xvalExp = Vector::Zero(4);

            auto result = optimizer.minimize(initGuess);

            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1);
            REQUIRE_MATRIX_APPROX(fvalExp, result.fval, 1);
            REQUIRE(Approx(Scalar{0}).margin(0.1) == result.error);
        }

        SECTION("barzilai borwein direct")
        {
            GradientDescentX<Scalar, ParabolicErrorNoJacobian, BarzilaiBorwein> optimizer;

            optimizer.setRefinementParameters({BarzilaiBorwein::Mode::Direct, static_cast<Scalar>(1e-2)});
            optimizer.setMinimumStepLength(static_cast<Scalar>(1e-10));
            optimizer.setMinimumGradientLength(static_cast<Scalar>(1e-10));
            optimizer.setMaximumIterations(100);

            Vector initGuess(4);
            initGuess << 2, 1, 3, 4;

            Vector fvalExp = Vector::Zero(2);
            Vector xvalExp = Vector::Zero(4);

            auto result = optimizer.minimize(initGuess);

            REQUIRE(result.converged);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
            REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE(Approx(Scalar{0}).margin(eps) == result.error);
        }

        SECTION("barzilai borwein inverse")
        {
            GradientDescentX<Scalar, ParabolicErrorNoJacobian, BarzilaiBorwein> optimizer;

            optimizer.setRefinementParameters({BarzilaiBorwein::Mode::Inverse, static_cast<Scalar>(1e-2)});
            optimizer.setMinimumStepLength(static_cast<Scalar>(1e-10));
            optimizer.setMinimumGradientLength(static_cast<Scalar>(1e-10));
            optimizer.setMaximumIterations(100);

            Vector initGuess(4);
            initGuess << 2, 1, 3, 4;

            Vector fvalExp = Vector::Zero(2);
            Vector xvalExp = Vector::Zero(4);

            auto result = optimizer.minimize(initGuess);

            REQUIRE(result.converged);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
            REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE(Approx(Scalar{0}).margin(eps) == result.error);
        }
    }
}