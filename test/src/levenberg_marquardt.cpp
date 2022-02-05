/// optimization.cpp
///
/// Author: Fabian Meyer
/// Created On: 05 Aug 2019


#include <lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsq;

TEST_CASE("levenberg marquardt")
{
    SECTION("double")
    {
        typedef double Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        const Scalar eps = 1e-3;

        SECTION("with jacobian")
        {
            LevenbergMarquardt<Scalar, ParabolicError<Scalar>> optimizer;

            optimizer.setMaxIterations(100);
            optimizer.setMinStepLength(1e-10);
            optimizer.setMinGradientLength(1e-10);

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
        }

        SECTION("without jacobian")
        {
            LevenbergMarquardt<Scalar, ParabolicErrorNoJacobian<Scalar>> optimizer;

            optimizer.setMaxIterations(100);
            optimizer.setMinStepLength(1e-10);
            optimizer.setMinGradientLength(1e-10);

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
        }

        SECTION("inverse jacobian")
        {
            LevenbergMarquardt<Scalar, ParabolicErrorInverseJacobian<Scalar>> optimizer;

            optimizer.setMaxIterations(100);
            optimizer.setMinStepLength(1e-10);
            optimizer.setMinGradientLength(1e-10);

            Vector initGuess(4);
            initGuess << 2, 1, 3, 4;

            Scalar errorExp = 0;
            Vector fvalExp = Vector::Zero(2);
            Vector xvalExp = Vector::Zero(4);

            auto result = optimizer.minimize(initGuess);

            REQUIRE_NOT_MATRIX_APPROX(xvalExp, result.xval, eps);
            REQUIRE_NOT_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE_FALSE(Approx(errorExp).margin(eps) == result.error);
        }
    }

    SECTION("float")
    {
        typedef float Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        const Scalar eps = 1e-3f;

        SECTION("with jacobian")
        {
            LevenbergMarquardt<Scalar, ParabolicError<Scalar>> optimizer;

            optimizer.setMaxIterations(100);
            optimizer.setMinStepLength(1e-10f);
            optimizer.setMinGradientLength(1e-10f);

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
        }

        SECTION("without jacobian")
        {
            LevenbergMarquardt<Scalar, ParabolicErrorNoJacobian<Scalar>> optimizer;

            optimizer.setMaxIterations(100);
            optimizer.setMinStepLength(1e-10f);
            optimizer.setMinGradientLength(1e-10f);

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
        }

        SECTION("inverse jacobian")
        {
            LevenbergMarquardt<Scalar, ParabolicErrorInverseJacobian<Scalar>> optimizer;

            optimizer.setMaxIterations(100);
            optimizer.setMinStepLength(1e-10f);
            optimizer.setMinGradientLength(1e-10f);

            Vector initGuess(4);
            initGuess << 2, 1, 3, 4;

            Scalar errorExp = 0;
            Vector fvalExp = Vector::Zero(2);
            Vector xvalExp = Vector::Zero(4);

            auto result = optimizer.minimize(initGuess);

            REQUIRE_NOT_MATRIX_APPROX(xvalExp, result.xval, eps);
            REQUIRE_NOT_MATRIX_APPROX(fvalExp, result.fval, eps);
            REQUIRE_FALSE(Approx(errorExp).margin(eps) == result.error);
        }
    }
}
