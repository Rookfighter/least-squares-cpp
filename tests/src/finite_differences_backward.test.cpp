/// finite_differences_backward.test.cpp
///
/// Author:     Fabian Meyer
/// Created On: 11 Nov 2020
/// License:    MIT

#include <lsqcpp/lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsqcpp;

TEMPLATE_TEST_CASE("backward differences", "[finite differences]", float, double)
{
    using Scalar = TestType;

    SECTION("construction")
    {
        SECTION("default")
        {
            FiniteDifferences<Scalar, BackwardDifferences> differences;
            REQUIRE(differences.epsilon() == std::sqrt(Eigen::NumTraits<Scalar>::epsilon()));
            REQUIRE(differences.threads() == 1);
        }

        SECTION("parametrized A")
        {
            FiniteDifferences<Scalar, BackwardDifferences> differences(13, 4);
            REQUIRE(differences.epsilon() == static_cast<Scalar>(13));
            REQUIRE(differences.threads() == 4);
        }

        SECTION("parametrized B")
        {
            FiniteDifferences<Scalar, BackwardDifferences> differences(21);
            REQUIRE(differences.epsilon() == static_cast<Scalar>(21));
            REQUIRE(differences.threads() == 1);
        }
    }

    SECTION("setter")
    {
        SECTION("epsilon")
        {
            FiniteDifferences<Scalar, BackwardDifferences> differences;
            REQUIRE(differences.epsilon() == std::sqrt(Eigen::NumTraits<Scalar>::epsilon()));

            differences.setEpsilon(4);
            REQUIRE(differences.epsilon() == static_cast<Scalar>(4));
        }

        SECTION("threads")
        {
            FiniteDifferences<Scalar, BackwardDifferences> differences;
            REQUIRE(differences.threads() == 1);

            differences.setThreads(11);
            REQUIRE(differences.threads() == 11);
        }
    }

    SECTION("jacobian")
    {
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
            FiniteDifferences<Scalar, BackwardDifferences> differences;

            const auto parabError = ParabolicError{};
            parabError(xval, fval, expected);
            differences(xval, fval, objective, actual);

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
            InputVector xval;
            xval <<
                static_cast<Scalar>(2.1),
                static_cast<Scalar>(1.7),
                static_cast<Scalar>(3.5),
                static_cast<Scalar>(5.9);
            OutputVector fval;
            FiniteDifferences<Scalar, BackwardDifferences> differences;

            const auto parabError = ParabolicError{};
            parabError(xval, fval, expected);
            differences(xval, fval, objective, actual);

            REQUIRE_MATRIX_APPROX(expected, actual, eps);
        }
    }
}
