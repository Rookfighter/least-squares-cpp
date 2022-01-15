/* dogleg_method.cpp
 *
 * Author: Fabian Meyer
 * Created On: 20 Nov 2020
 */

#include <lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsq;

TEST_CASE("dogleg method" DATATYPE_STR)
{
    typedef DATATYPE Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    const Scalar eps = static_cast<Scalar>(1e-3);

    SECTION("with jacobian")
    {
        DoglegMethod<Scalar, ParabolicError<Scalar>> optimizer;

        optimizer.setMinStepLength(static_cast<Scalar>(1e-10));
        optimizer.setMinGradientLength(static_cast<Scalar>(1e-10));
        optimizer.setMaxIterations(100);
        optimizer.setMaxIterationsTR(100);

        Vector initGuess(4);
        initGuess <<
            static_cast<Scalar>(2),
            static_cast<Scalar>(1),
            static_cast<Scalar>(3),
            static_cast<Scalar>(4);

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
        DoglegMethod<Scalar, ParabolicErrorNoJacobian<Scalar>> optimizer;

        optimizer.setMinStepLength(static_cast<Scalar>(1e-10));
        optimizer.setMinGradientLength(static_cast<Scalar>(1e-10));
        optimizer.setMaxIterations(100);
        optimizer.setMaxIterationsTR(100);

        Vector initGuess(4);
        initGuess <<
            static_cast<Scalar>(2),
            static_cast<Scalar>(1),
            static_cast<Scalar>(3),
            static_cast<Scalar>(4);

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
