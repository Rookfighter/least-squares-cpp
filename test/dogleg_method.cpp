/* optimization.cpp
 *
 * Author: Fabian Meyer
 * Created On: 05 Aug 2019
 */

#include <lsqcpp.h>
#include "assert/eigen_require.h"
#include "errors/parabolic_error.h"

using namespace lsq;

TEST_CASE("dogleg method")
{
    SECTION("double")
    {
        typedef double Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        const Scalar eps = static_cast<Scalar>(1e-3);

        SECTION("with jacobian")
        {
            DoglegMethod<Scalar, ParabolicError<Scalar>> optimizer;

            optimizer.setMinStepLength(1e-10);
            optimizer.setMinGradientLength(1e-10);
            optimizer.setMaxIterations(100);
            optimizer.setMaxIterationsTR(100);

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

        SECTION("without jacobian")
        {
             DoglegMethod<Scalar, ParabolicErrorNoJacobian<Scalar>> optimizer;

            optimizer.setMinStepLength(1e-10);
            optimizer.setMinGradientLength(1e-10);
            optimizer.setMaxIterations(100);
            optimizer.setMaxIterationsTR(100);

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
    }
}
