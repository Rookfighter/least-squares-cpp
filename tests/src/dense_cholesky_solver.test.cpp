/// dense_cholesky_solver.test.cpp
///
/// Author:     Fabian Meyer
/// Created On: 05 Aug 2019
/// License:    MIT

#include <lsqcpp/lsqcpp.h>
#include "eigen_require.h"

using namespace lsq;

TEMPLATE_TEST_CASE("dense cholesky solver", "[dense solver]", float, double)
{
    using Scalar = TestType;
    const Scalar eps = static_cast<Scalar>(1e-6);

    SECTION("dynamic size")
    {
        using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        SECTION("deny non positive semi-definite")
        {
            Matrix A(4, 4);
            A << 2, 3, 11, 5,
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

            Vector actual;
            DenseCholeskySolver solver;
            REQUIRE(!solver(A, b, actual));
        }

        SECTION("solve positive semi-definite")
        {
            Matrix A(4, 4);
            A << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;
            Vector b(4);
            b << 2, 1, 4, 5;

            Vector expected(4);
            expected << 2, 1, 4, 5;

            Vector actual;
            DenseCholeskySolver solver;
            REQUIRE(solver(A, b, actual));
            REQUIRE_MATRIX_APPROX(expected, actual, eps);
        }
    }

    SECTION("fixed size")
    {
        using Vector = Eigen::Matrix<Scalar, 4, 1>;
        using Matrix = Eigen::Matrix<Scalar, 4, 4>;

        SECTION("deny non positive semi-definite")
        {
            Matrix A;
            A << 2, 3, 11, 5,
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

            Vector actual;
            DenseCholeskySolver solver;
            REQUIRE(!solver(A, b, actual));
        }

        SECTION("solve positive semi-definite")
        {
            Matrix A;
            A << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;
            Vector b;
            b << 2, 1, 4, 5;

            Vector expected;
            expected << 2, 1, 4, 5;

            Vector actual;
            DenseCholeskySolver solver;
            REQUIRE(solver(A, b, actual));
            REQUIRE_MATRIX_APPROX(expected, actual, eps);
        }
    }
}
