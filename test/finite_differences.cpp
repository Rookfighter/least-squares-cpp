/* finite_differences.cpp
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

std::function<void(const Vector &, Vector &)> errorFunction =
    [](const Vector &xval, Vector &fval)
    {
        ParabolicError<Scalar> parabError;
        Matrix jacobian;
        parabError(xval, fval, jacobian);
    };

TEST_CASE("forward differences")
{
    const Scalar eps = 1e-6;
    Matrix jacobianAct;
    Matrix jacobianExp;
    ParabolicError<Scalar> parabError;
    Vector xval(4);
    xval << 2.1, 1.7, 3.5, 5.9;
    Vector fval;
    ForwardDifferences<Scalar> differences;

    differences.setErrorFunction(errorFunction);

    parabError(xval, fval, jacobianExp);
    differences(xval, fval, jacobianAct);

    REQUIRE_MATRIX_APPROX(jacobianExp, jacobianAct, eps);

}

TEST_CASE("backward differences")
{
    const Scalar eps = 1e-6;
    Matrix jacobianAct;
    Matrix jacobianExp;
    ParabolicError<Scalar> parabError;
    Vector xval(4);
    xval << 2.1, 1.7, 3.5, 5.9;
    Vector fval;
    BackwardDifferences<Scalar> differences;

    differences.setErrorFunction(errorFunction);

    parabError(xval, fval, jacobianExp);
    differences(xval, fval, jacobianAct);

    REQUIRE_MATRIX_APPROX(jacobianExp, jacobianAct, eps);
}

TEST_CASE("central differences")
{
    const Scalar eps = 1e-6;
    Matrix jacobianAct;
    Matrix jacobianExp;
    ParabolicError<Scalar> parabError;
    Vector xval(4);
    xval << 2.1, 1.7, 3.5, 5.9;
    Vector fval;
    CentralDifferences<Scalar> differences;

    differences.setErrorFunction(errorFunction);

    parabError(xval, fval, jacobianExp);
    differences(xval, fval, jacobianAct);

    REQUIRE_MATRIX_APPROX(jacobianExp, jacobianAct, eps);
}
