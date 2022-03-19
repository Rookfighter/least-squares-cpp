/// optimization.cpp
///
/// Author: Fabian Meyer
/// Created On: 05 Aug 2019


#include <lsqcpp/lsqcpp.hpp>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsqcpp;


TEST_CASE("gradient descent")
{
    SECTION("double")
    {
        typedef double Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        const Scalar eps = 1e-3;

        SECTION("with jacobian")
        {
            SECTION("constant step size")
            {
                GradientDescent<Scalar, ParabolicError<Scalar>, ConstantStepSize<Scalar>> optimizer;

                optimizer.setStepSize({1e-2});
                optimizer.setMinStepLength(1e-10);
                optimizer.setMinGradientLength(1e-10);
                optimizer.setMaxIterations(100);

                Vector initGuess(4);
                initGuess << 2, 1, 3, 4;

                Scalar errorExp = 0;
                Vector fvalExp = Vector::Zero(2);
                Vector xvalExp = Vector::Zero(4);

                auto result = optimizer.minimize(initGuess);

                REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1);
                REQUIRE_MATRIX_APPROX(fvalExp, result.fval, 1);
                REQUIRE(Approx(errorExp).margin(0.1) == result.error);
            }

            SECTION("barzilai borwein direct")
            {
                GradientDescent<Scalar, ParabolicError<Scalar>, BarzilaiBorwein<Scalar>> optimizer;

                optimizer.setStepSize({BarzilaiBorwein<Scalar>::Method::Direct, 1e-2});
                optimizer.setMinStepLength(1e-10);
                optimizer.setMinGradientLength(1e-10);
                optimizer.setMaxIterations(100);

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

            SECTION("barzilai borwein inverse")
            {
                GradientDescent<Scalar, ParabolicError<Scalar>, BarzilaiBorwein<Scalar>> optimizer;

                optimizer.setStepSize({BarzilaiBorwein<Scalar>::Method::Inverse, 1e-2});
                optimizer.setMinStepLength(1e-10);
                optimizer.setMinGradientLength(1e-10);
                optimizer.setMaxIterations(100);

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
        }

        SECTION("without jacobian")
        {
            SECTION("constant step size")
            {
                GradientDescent<Scalar, ParabolicErrorNoJacobian<Scalar>, ConstantStepSize<Scalar>> optimizer;

                optimizer.setStepSize({1e-2});
                optimizer.setMinStepLength(1e-10);
                optimizer.setMinGradientLength(1e-10);
                optimizer.setMaxIterations(100);

                Vector initGuess(4);
                initGuess << 2, 1, 3, 4;

                Scalar errorExp = 0;
                Vector fvalExp = Vector::Zero(2);
                Vector xvalExp = Vector::Zero(4);

                auto result = optimizer.minimize(initGuess);

                REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1);
                REQUIRE_MATRIX_APPROX(fvalExp, result.fval, 1);
                REQUIRE(Approx(errorExp).margin(0.1) == result.error);
            }

            SECTION("barzilai borwein direct")
            {
                GradientDescent<Scalar, ParabolicErrorNoJacobian<Scalar>, BarzilaiBorwein<Scalar>> optimizer;

                optimizer.setStepSize({BarzilaiBorwein<Scalar>::Method::Direct, 1e-2});
                optimizer.setMinStepLength(1e-10);
                optimizer.setMinGradientLength(1e-10);
                optimizer.setMaxIterations(100);

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

            SECTION("barzilai borwein inverse")
            {
                GradientDescent<Scalar, ParabolicErrorNoJacobian<Scalar>, BarzilaiBorwein<Scalar>> optimizer;

                optimizer.setStepSize({BarzilaiBorwein<Scalar>::Method::Inverse, 1e-2});
                optimizer.setMinStepLength(1e-10);
                optimizer.setMinGradientLength(1e-10);
                optimizer.setMaxIterations(100);

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
        }
    }

    SECTION("float")
    {
        typedef float Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        const Scalar eps = 1e-3f;

        SECTION("with jacobian")
        {
            SECTION("constant step size")
            {

                GradientDescent<Scalar, ParabolicError<Scalar>, ConstantStepSize<Scalar>> optimizer;

                optimizer.setStepSize({1e-2f});
                optimizer.setMinStepLength(1e-10f);
                optimizer.setMinGradientLength(1e-10f);
                optimizer.setMaxIterations(100);

                Vector initGuess(4);
                initGuess << 2, 1, 3, 4;

                Scalar errorExp = 0;
                Vector fvalExp = Vector::Zero(2);
                Vector xvalExp = Vector::Zero(4);

                auto result = optimizer.minimize(initGuess);

                REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1.0f);
                REQUIRE_MATRIX_APPROX(fvalExp, result.fval, 1.0f);
                REQUIRE(Approx(errorExp).margin(0.1) == result.error);
            }

            SECTION("barzilai borwein direct")
            {

                GradientDescent<Scalar, ParabolicError<Scalar>, BarzilaiBorwein<Scalar>> optimizer;

                optimizer.setStepSize({BarzilaiBorwein<Scalar>::Method::Direct, 1e-2f});
                optimizer.setMinStepLength(1e-10f);
                optimizer.setMinGradientLength(1e-10f);
                optimizer.setMaxIterations(100);

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

            SECTION("barzilai borwein inverse")
            {
                GradientDescent<Scalar, ParabolicError<Scalar>, BarzilaiBorwein<Scalar>> optimizer;

                optimizer.setStepSize({BarzilaiBorwein<Scalar>::Method::Inverse, 1e-2f});
                optimizer.setMinStepLength(1e-10f);
                optimizer.setMinGradientLength(1e-10f);
                optimizer.setMaxIterations(100);

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
        }

        SECTION("without jacobian")
        {
            SECTION("constant step size")
            {
                GradientDescent<Scalar, ParabolicErrorNoJacobian<Scalar>, ConstantStepSize<Scalar>> optimizer;

                optimizer.setStepSize({1e-2f});
                optimizer.setMinStepLength(1e-10f);
                optimizer.setMinGradientLength(1e-10f);
                optimizer.setMaxIterations(100);

                Vector initGuess(4);
                initGuess << 2, 1, 3, 4;

                Scalar errorExp = 0;
                Vector fvalExp = Vector::Zero(2);
                Vector xvalExp = Vector::Zero(4);

                auto result = optimizer.minimize(initGuess);

                REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1);
                REQUIRE_MATRIX_APPROX(fvalExp, result.fval, 1);
                REQUIRE(Approx(errorExp).margin(0.1) == result.error);
            }

            SECTION("barzilai borwein direct")
            {
                GradientDescent<Scalar, ParabolicErrorNoJacobian<Scalar>, BarzilaiBorwein<Scalar>> optimizer;

                optimizer.setStepSize({BarzilaiBorwein<Scalar>::Method::Direct, 1e-2f});
                optimizer.setMinStepLength(1e-10f);
                optimizer.setMinGradientLength(1e-10f);
                optimizer.setMaxIterations(100);

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

            SECTION("barzilai borwein inverse")
            {
                GradientDescent<Scalar, ParabolicErrorNoJacobian<Scalar>, BarzilaiBorwein<Scalar>> optimizer;

                optimizer.setStepSize({BarzilaiBorwein<Scalar>::Method::Inverse, 1e-2f});
                optimizer.setMinStepLength(1e-10f);
                optimizer.setMinGradientLength(1e-10f);
                optimizer.setMaxIterations(100);

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
        }
    }
}