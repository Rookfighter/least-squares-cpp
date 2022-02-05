/// dense_svd_solver.cpp
///
/// Author: Fabian Meyer
/// Created On: 05 Aug 2019


#include <lsqcpp.h>
#include "eigen_require.h"

using namespace lsq;


TEMPLATE_TEST_CASE("dense SVD solver", "[dense solver]", float, double)
{
    using Scalar = TestType;
    const Scalar eps = static_cast<Scalar>(1e-6);

    SECTION("dynamic size")
    {
        using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        Matrix A(4, 4);
        A <<
            2, 3, 11, 5,
            1, 1, 5, 2,
            2, 1, -3, 2,
            1, 1, -3, 4;
        Vector b(4);
        b << 2, 1, -3, -3;

        Vector expected(4);
        expected <<
            static_cast<Scalar>(-0.5),
            static_cast<Scalar>(-0.1875),
            static_cast<Scalar>(0.4375),
            static_cast<Scalar>(-0.25);

        DenseSVDSolver solver;
        Vector actual = solver(A, b);

        REQUIRE_MATRIX_APPROX(expected, actual, eps);
    }

    SECTION("fixed size")
    {
        using Vector = Eigen::Matrix<Scalar, 4, 1>;
        using Matrix = Eigen::Matrix<Scalar, 4, 4>;

        Matrix A;
        A <<
            2, 3, 11, 5,
            1, 1, 5, 2,
            2, 1, -3, 2,
            1, 1, -3, 4;
        Vector b;
        b << 2, 1, -3, -3;

        Vector expected;
        expected <<
            static_cast<Scalar>(-0.5),
            static_cast<Scalar>(-0.1875),
            static_cast<Scalar>(0.4375),
            static_cast<Scalar>(-0.25);

        DenseSVDSolver solver;
        Vector actual = solver(A, b);

        REQUIRE_MATRIX_APPROX(expected, actual, eps);
    }
}
