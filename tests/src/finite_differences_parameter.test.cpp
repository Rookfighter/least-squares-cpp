/// finite_differences_parameter.test.cpp
///
/// Author:     Fabian Meyer
/// Created On: 11 Nov 2020
/// License:    MIT

#include <lsqcpp/lsqcpp.hpp>
#include "eigen_require.hpp"

using namespace lsqcpp;

TEMPLATE_TEST_CASE("finite differences parameter", "[finite differences]", float, double)
{
    using Scalar = TestType;

    SECTION("construction")
    {
        SECTION("default")
        {
            FiniteDifferencesParameter<Scalar> param;
            REQUIRE(param.epsilon() == std::sqrt(Eigen::NumTraits<Scalar>::epsilon()));
            REQUIRE(param.threads() == 1);
        }

        SECTION("parametrized A")
        {
            FiniteDifferencesParameter<Scalar> param(13, 4);
            REQUIRE(param.epsilon() == static_cast<Scalar>(13));
            REQUIRE(param.threads() == 4);
        }

        SECTION("parametrized B")
        {
            FiniteDifferencesParameter<Scalar> param(21);
            REQUIRE(param.epsilon() == static_cast<Scalar>(21));
            REQUIRE(param.threads() == 1);
        }
    }

    SECTION("setter")
    {
        SECTION("epsilon")
        {
            FiniteDifferencesParameter<Scalar> param;
            REQUIRE(param.epsilon() == std::sqrt(Eigen::NumTraits<Scalar>::epsilon()));

            param.setEpsilon(4);
            REQUIRE(param.epsilon() == static_cast<Scalar>(4));
        }

        SECTION("threads")
        {
            FiniteDifferencesParameter<Scalar> param;
            REQUIRE(param.threads() == 1);

            param.setThreads(11);
            REQUIRE(param.threads() == 11);
        }
    }
}
