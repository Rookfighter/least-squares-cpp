/// finite_differences_forward.test.cpp
///
/// Author:     Fabian Meyer
/// Created On: 11 Nov 2020
/// License:    MIT

#include <lsqcpp/lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsqcpp;

TEMPLATE_TEST_CASE("forward differences", "[finite differences]", float, double)
{
    using Scalar = TestType;

    SECTION("dynamic size problem")
    {
        using InputVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using OutputVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using JacobiMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        constexpr auto eps = static_cast<Scalar>(1e-1);

        const auto objective = ParabolicErrorNoJacobian{};

        JacobiMatrix actual;
        JacobiMatrix expected;
        InputVector xval(4);
        xval <<
            static_cast<Scalar>(2.1),
            static_cast<Scalar>(1.7),
            static_cast<Scalar>(3.5),
            static_cast<Scalar>(5.9);
        OutputVector fval;

        ForwardDifferences differences;
        FiniteDifferencesParameter<Scalar> param;

        const auto parabError = ParabolicError{};
        parabError(xval, fval, expected);
        differences(xval, fval, objective, param, actual);

        REQUIRE_MATRIX_APPROX(expected, actual, eps);
    }

    SECTION("fixed size problem")
    {
        using InputVector = Eigen::Matrix<Scalar, 4, 1>;
        using OutputVector = Eigen::Matrix<Scalar, 2, 1>;
        using JacobiMatrix = Eigen::Matrix<Scalar, 2, 4>;
        constexpr auto eps = static_cast<Scalar>(1e-1);

        const auto objective = ParabolicErrorNoJacobian{};

        JacobiMatrix actual;
        JacobiMatrix expected;
        InputVector xval(4);
        xval <<
            static_cast<Scalar>(2.1),
            static_cast<Scalar>(1.7),
            static_cast<Scalar>(3.5),
            static_cast<Scalar>(5.9);
        OutputVector fval;

        ForwardDifferences differences;
        FiniteDifferencesParameter<Scalar> param;

        const auto parabError = ParabolicError{};
        parabError(xval, fval, expected);
        differences(xval, fval, objective, param, actual);

        REQUIRE_MATRIX_APPROX(expected, actual, eps);
    }
}