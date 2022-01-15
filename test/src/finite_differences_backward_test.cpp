/* finite_differences_backward_test.cpp
 *
 * Author: Fabian Meyer
 * Created On: 11 Nov 2020
 */

#include <lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsq;

TEMPLATE_TEST_CASE("backward differences", "[finite differences]", float, double)
{
    using Scalar = TestType;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    constexpr auto eps = static_cast<Scalar>(1e-1);

    const auto objective = ParabolicErrorNoJacobian<Scalar>();

    Matrix actual;
    Matrix expected;
    Vector xval(4);
    xval <<
        static_cast<Scalar>(2.1),
        static_cast<Scalar>(1.7),
        static_cast<Scalar>(3.5),
        static_cast<Scalar>(5.9);
    Vector fval;
    FiniteDifferences<Scalar, BackwardDifferences> differences;

    ParabolicError<Scalar> parabError;
    parabError(xval, fval, expected);
    differences(xval, fval, objective, actual);

    REQUIRE_MATRIX_APPROX(expected, actual, eps);
}
