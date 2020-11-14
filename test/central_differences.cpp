/* central_differences.cpp
 *
 * Author: Fabian Meyer
 * Created On: 11 Nov 2020
 */

#include <lsqcpp.h>
#include "assert/eigen_require.h"
#include "errors/parabolic_error.h"

using namespace lsq;

TEST_CASE("central differences")
{
    SECTION("double")
    {
        typedef double Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        const Scalar eps = static_cast<Scalar>(1e-6);

        Matrix jacobianAct;
        Matrix jacobianExp;
        ParabolicError<Scalar> parabError;
        Vector xval(4);
        xval << 2.1, 1.7, 3.5, 5.9;
        Vector fval;
        CentralDifferences<Scalar> differences;

        differences.setErrorFunction(
        [](const Vector &xval, Vector &fval)
        {
            ParabolicError<Scalar> parabError;
            Matrix jacobian;
            parabError(xval, fval, jacobian);
        });

        parabError(xval, fval, jacobianExp);
        differences(xval, fval, jacobianAct);

        REQUIRE_MATRIX_APPROX(jacobianExp, jacobianAct, eps);
    }

    SECTION("float")
    {
        typedef float Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        const Scalar eps = static_cast<Scalar>(1e-1);

        Matrix jacobianAct;
        Matrix jacobianExp;
        ParabolicError<Scalar> parabError;
        Vector xval(4);
        xval << 2.1f, 1.7f, 3.5f, 5.9f;
        Vector fval;
        CentralDifferences<Scalar> differences;

        differences.setErrorFunction(
        [](const Vector &xval, Vector &fval)
        {
            ParabolicError<Scalar> parabError;
            Matrix jacobian;
            parabError(xval, fval, jacobian);
        });

        parabError(xval, fval, jacobianExp);
        differences(xval, fval, jacobianAct);

        REQUIRE_MATRIX_APPROX(jacobianExp, jacobianAct, eps);
    }
}