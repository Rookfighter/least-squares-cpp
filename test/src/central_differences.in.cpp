/* central_differences.cpp
 *
 * Author: Fabian Meyer
 * Created On: 11 Nov 2020
 */

#include <lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsq;

TEST_CASE("central differences " DATATYPE_STR)
{
    typedef DATATYPE Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    const Scalar eps = static_cast<Scalar>(1e-1);

    Matrix jacobianAct;
    Matrix jacobianExp;
    ParabolicError<Scalar> parabError;
    Vector xval(4);
    xval <<
        static_cast<Scalar>(2.1),
        static_cast<Scalar>(1.7),
        static_cast<Scalar>(3.5),
        static_cast<Scalar>(5.9);

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