/* optimization.cpp
 *
 * Author: Fabian Meyer
 * Created On: 05 Aug 2019
 */

#include <lsqcpp.h>
#include "assert/eigen_require.h"

using namespace lsq;

template<typename Scalar>
struct ParabolicError
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    void operator()(const Vector &xval, Vector &fval, Matrix &jacobian)
    {
        assert(xval.size() % 2 == 0);

        fval.resize(xval.size() / 2);
        jacobian.setZero(fval.size(), xval.size());

        for(Index i = 0; i < fval.size(); ++i)
        {
            fval(i) = xval(i*2) * xval(i*2) + xval(i*2+1) * xval(i*2+1);
            jacobian(i, i*2) = 2 * xval(i*2);
            jacobian(i, i*2+1) = 2 * xval(i*2+1);
        }
    }
};

typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

TEST_CASE("gradient descent")
{
    const Scalar eps = 1e-3;

    GradientDescent<Scalar, ParabolicError<Scalar>> optimizer;

    optimizer.setMinStepLength(1e-10);
    optimizer.setMinGradientLength(1e-10);
    optimizer.setMaxIterations(100);

    Vector initGuess(4);
    initGuess << 2, 1, 3, 4;

    Scalar errorExp = 0;
    Vector fvalExp = Vector::Zero(2);
    Vector xvalExp = Vector::Zero(4);

    auto result = optimizer.minimize(initGuess);

    REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
    REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
    REQUIRE(Approx(errorExp).margin(eps) == result.error);
}

TEST_CASE("gauss newton")
{
    const Scalar eps = 1e-3;

    GaussNewton<Scalar, ParabolicError<Scalar>> optimizer;

    optimizer.setStepSize({0.8, 0.1, 1e-10, 1.0, 50});
    optimizer.setMinStepLength(1e-10);
    optimizer.setMinGradientLength(1e-10);
    optimizer.setMaxIterations(100);

    Vector initGuess(4);
    initGuess << 2, 1, 3, 4;

    Scalar errorExp = 0;
    Vector fvalExp = Vector::Zero(2);
    Vector xvalExp = Vector::Zero(4);

    auto result = optimizer.minimize(initGuess);

    REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
    REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
    REQUIRE(Approx(errorExp).margin(eps) == result.error);
}

TEST_CASE("levenberg marquardt")
{
    const Scalar eps = 1e-3;

    LevenbergMarquardt<Scalar, ParabolicError<Scalar>> optimizer;

    optimizer.setStepSize({1.0});
    optimizer.setMaxIterations(100);
    optimizer.setMinStepLength(1e-10);
    optimizer.setMinGradientLength(1e-10);

    Vector initGuess(4);
    initGuess << 2, 1, 3, 4;

    Scalar errorExp = 0;
    Vector fvalExp = Vector::Zero(2);
    Vector xvalExp = Vector::Zero(4);

    auto result = optimizer.minimize(initGuess);

    REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
    REQUIRE_MATRIX_APPROX(fvalExp, result.fval, eps);
    REQUIRE(Approx(errorExp).margin(eps) == result.error);
}
