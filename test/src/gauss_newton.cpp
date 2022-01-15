/* optimization.cpp
 *
 * Author: Fabian Meyer
 * Created On: 05 Aug 2019
 */

#include <lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsq;

TEST_CASE("gauss newton")
{
    SECTION("double")
    {
        typedef double Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        const Scalar eps = static_cast<Scalar>(1e-3);

        SECTION("with jacobian")
        {
            SECTION("constant step size converge")
            {
                GaussNewton<Scalar, ParabolicError<Scalar>, ConstantStepSize<Scalar>> optimizer;

                optimizer.setStepSize({0.5});
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
                REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1e-1);
                REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
            }

            SECTION("armijo backtracking")
            {
                GaussNewton<Scalar, ParabolicError<Scalar>, ArmijoBacktracking<Scalar>> optimizer;

                optimizer.setStepSize({0.8, 1e-4, 1e-10, 1.0, 50});
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
                REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1e-1);
                REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
            }

            SECTION("wolfe backtracking")
            {
                GaussNewton<Scalar, ParabolicError<Scalar>, WolfeBacktracking<Scalar>> optimizer;

                optimizer.setStepSize({0.8, 1e-4, 0.1, 1e-10, 1.0, 100});
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
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
            }
        }

        SECTION("without jacobian")
        {
            SECTION("constant step size converge")
            {
                GaussNewton<Scalar, ParabolicErrorNoJacobian<Scalar>, ConstantStepSize<Scalar>> optimizer;

                optimizer.setStepSize({0.5});
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
                REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1e-1);
                REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
            }

            SECTION("armijo backtracking")
            {
                GaussNewton<Scalar, ParabolicErrorNoJacobian<Scalar>, ArmijoBacktracking<Scalar>> optimizer;

                optimizer.setStepSize({0.8, 1e-4, 1e-10, 1.0, 50});
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
                REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1e-1);
                REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
            }

            SECTION("wolfe backtracking")
            {
                GaussNewton<Scalar, ParabolicErrorNoJacobian<Scalar>, WolfeBacktracking<Scalar>> optimizer;

                optimizer.setStepSize({0.8, 1e-4, 0.1, 1e-10, 1.0, 100});
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
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
            }
        }

        SECTION("inverse jacobian")
        {
            SECTION("constant step size converge")
            {
                GaussNewton<Scalar, ParabolicErrorInverseJacobian<Scalar>, ConstantStepSize<Scalar>> optimizer;

                optimizer.setStepSize({0.5});
                optimizer.setMinStepLength(1e-10);
                optimizer.setMinGradientLength(1e-10);
                optimizer.setMaxIterations(100);

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
                GaussNewton<Scalar, ParabolicErrorInverseJacobian<Scalar>, ArmijoBacktracking<Scalar>> optimizer;

                optimizer.setStepSize({0.8, 1e-4, 1e-10, 1.0, 50});
                optimizer.setMinStepLength(1e-10);
                optimizer.setMinGradientLength(1e-10);
                optimizer.setMaxIterations(100);

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
                GaussNewton<Scalar, ParabolicErrorInverseJacobian<Scalar>, WolfeBacktracking<Scalar>> optimizer;

                optimizer.setStepSize({0.8, 1e-4, 0.1, 1e-10, 1.0, 100});
                optimizer.setMinStepLength(1e-10);
                optimizer.setMinGradientLength(1e-10);
                optimizer.setMaxIterations(100);

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

    SECTION("float")
    {
        typedef float Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        const Scalar eps = static_cast<Scalar>(1e-3);

        SECTION("with jacobian")
        {
            SECTION("constant step size converge")
            {
                GaussNewton<Scalar, ParabolicError<Scalar>, ConstantStepSize<Scalar>> optimizer;

                optimizer.setStepSize({0.5f});
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
                REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1e-1f);
                REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
            }

            SECTION("armijo backtracking")
            {
                GaussNewton<Scalar, ParabolicError<Scalar>, ArmijoBacktracking<Scalar>> optimizer;

                optimizer.setStepSize({0.8f, 1e-4f, 1e-10f, 1.0f, 50});
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
                REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1e-1f);
                REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
            }

            SECTION("wolfe backtracking")
            {
                GaussNewton<Scalar, ParabolicError<Scalar>, WolfeBacktracking<Scalar>> optimizer;

                optimizer.setStepSize({0.8f, 1e-4f, 0.1f, 1e-10f, 1.0f, 100});
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
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
            }
        }

        SECTION("without jacobian")
        {
            SECTION("constant step size converge")
            {
                GaussNewton<Scalar, ParabolicErrorNoJacobian<Scalar>, ConstantStepSize<Scalar>> optimizer;

                optimizer.setStepSize({0.5f});
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
                REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1e-1f);
                REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
            }

            SECTION("armijo backtracking")
            {
                GaussNewton<Scalar, ParabolicErrorNoJacobian<Scalar>, ArmijoBacktracking<Scalar>> optimizer;

                optimizer.setStepSize({0.8f, 1e-4f, 1e-10f, 1.0f, 50});
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
                REQUIRE_MATRIX_APPROX(xvalExp, result.xval, 1e-1f);
                REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
            }

            SECTION("wolfe backtracking")
            {
                GaussNewton<Scalar, ParabolicErrorNoJacobian<Scalar>, WolfeBacktracking<Scalar>> optimizer;

                optimizer.setStepSize({0.8f, 1e-4f, 0.1f, 1e-10f, 1.0f, 100});
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
                REQUIRE(Approx(errorExp).margin(eps) == result.error);
            }
        }

        SECTION("inverse jacobian")
        {
            SECTION("constant step size converge")
            {
                GaussNewton<Scalar, ParabolicErrorInverseJacobian<Scalar>, ConstantStepSize<Scalar>> optimizer;

                optimizer.setStepSize({0.5f});
                optimizer.setMinStepLength(1e-10f);
                optimizer.setMinGradientLength(1e-10f);
                optimizer.setMaxIterations(100);

                Vector initGuess(4);
                initGuess << 2, 1, 3, 4;

                Scalar errorExp = 0;
                Vector fvalExp = Vector::Zero(2);
                Vector xvalExp = Vector::Zero(4);

                auto result = optimizer.minimize(initGuess);

                REQUIRE_FALSE(result.converged);
                REQUIRE_NOT_MATRIX_APPROX(xvalExp, result.xval, eps);
                REQUIRE_NOT_MATRIX_APPROX(fvalExp, result.fval, eps);
                REQUIRE_FALSE(Approx(errorExp).margin(eps) == result.error);
                REQUIRE_FALSE(Approx(errorExp).margin(eps) == result.error);
            }

            SECTION("armijo backtracking")
            {
                GaussNewton<Scalar, ParabolicErrorInverseJacobian<Scalar>, ArmijoBacktracking<Scalar>> optimizer;

                optimizer.setStepSize({0.8f, 1e-4f, 1e-10f, 1.0f, 50});
                optimizer.setMinStepLength(1e-10f);
                optimizer.setMinGradientLength(1e-10f);
                optimizer.setMaxIterations(100);

                Vector initGuess(4);
                initGuess << 2, 1, 3, 4;

                Scalar errorExp = 0;
                Vector fvalExp = Vector::Zero(2);
                Vector xvalExp = Vector::Zero(4);

                auto result = optimizer.minimize(initGuess);

                REQUIRE_FALSE(result.converged);
                REQUIRE_NOT_MATRIX_APPROX(xvalExp, result.xval, eps);
                REQUIRE_NOT_MATRIX_APPROX(fvalExp, result.fval, eps);
                REQUIRE_FALSE(Approx(errorExp).margin(eps) == result.error);
                REQUIRE_FALSE(Approx(errorExp).margin(eps) == result.error);
            }

            SECTION("wolfe backtracking")
            {
                GaussNewton<Scalar, ParabolicErrorInverseJacobian<Scalar>, WolfeBacktracking<Scalar>> optimizer;

                optimizer.setStepSize({0.8f, 1e-4f, 0.1f, 1e-10f, 1.0f, 100});
                optimizer.setMinStepLength(1e-10f);
                optimizer.setMinGradientLength(1e-10f);
                optimizer.setMaxIterations(100);

                Vector initGuess(4);
                initGuess << 2, 1, 3, 4;

                Scalar errorExp = 0;
                Vector fvalExp = Vector::Zero(2);
                Vector xvalExp = Vector::Zero(4);

                auto result = optimizer.minimize(initGuess);

                REQUIRE_FALSE(result.converged);
                REQUIRE_NOT_MATRIX_APPROX(xvalExp, result.xval, eps);
                REQUIRE_NOT_MATRIX_APPROX(fvalExp, result.fval, eps);
                REQUIRE_FALSE(Approx(errorExp).margin(eps) == result.error);
                REQUIRE_FALSE(Approx(errorExp).margin(eps) == result.error);
            }
        }
    }
}
