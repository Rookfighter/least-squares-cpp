/* dense_solver.cpp
 *
 * Author: Fabian Meyer
 * Created On: 05 Aug 2019
 */

#include <lsqcpp.h>
#include "assert/eigen_require.h"

using namespace lsq;

typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

TEST_CASE("dense SVD solver")
{
    const Scalar eps = 1e-6;

    Matrix A(4, 4);
    A << 2, 3, 11, 5,
        1, 1, 5, 2,
        2, 1, -3, 2,
        1, 1, -3, 4;
    Vector b(4);
    b << 2, 1, -3, -3;

    Vector resultAct;
    Vector resultExp(4);
    resultExp << -0.5, -0.1875, 0.4375, -0.25;

    DenseSVDSolver<Scalar> solver;
    solver(A, b, resultAct);

    REQUIRE_MATRIX_APPROX(resultExp, resultAct, eps);
}

TEST_CASE("dense Cholesky solver")
{
    const Scalar eps = 1e-6;

    SECTION("deny non positive semi-definite")
    {
        Matrix A(4, 4);
        A << 2, 3, 11, 5,
            1, 1, 5, 2,
            2, 1, -3, 2,
            1, 1, -3, 4;
        Vector b(4);
        b << 2, 1, -3, -3;

        Vector resultAct;
        Vector resultExp(4);
        resultExp << -0.5, -0.1875, 0.4375, -0.25;

        DenseCholeskySolver<Scalar> solver;
        REQUIRE_THROWS(solver(A, b, resultAct));
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

        Vector resultAct;
        Vector resultExp(4);
        resultExp << 2, 1, 4, 5;

        DenseCholeskySolver<Scalar> solver;
        solver(A, b, resultAct);

        REQUIRE_MATRIX_APPROX(resultExp, resultAct, eps);
    }
}
