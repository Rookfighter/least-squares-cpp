/// step_refiner_dogleg_method.test.cpp
///
/// Author:     Fabian Meyer
/// Created On: 22 Jan 2021
/// License:    MIT

#include <lsqcpp/lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsqcpp;

TEMPLATE_TEST_CASE("dogleg method step refiner ", "[step refiner]", float, double)
{
    using Scalar = TestType;
    using Parameter = DoglegMethodParameter<Scalar>;

    SECTION("parameter")
    {
        SECTION("construction")
        {
            SECTION("default")
            {
                Parameter param;
                REQUIRE(param.initialRadius() == static_cast<Scalar>(1));
                REQUIRE(param.maximumRadius() == static_cast<Scalar>(2));
                REQUIRE(param.radiusEpsilon() == static_cast<Scalar>(1e-6));
                REQUIRE(param.acceptanceFitness() == static_cast<Scalar>(0.25));
                REQUIRE(param.maximumIterations() == 0);
            }

            SECTION("parametrized A")
            {
                Parameter param(static_cast<Scalar>(0.42),
                                static_cast<Scalar>(0.78),
                                static_cast<Scalar>(1e-2),
                                static_cast<Scalar>(0.11),
                                10);
                REQUIRE(param.initialRadius() == static_cast<Scalar>(0.42));
                REQUIRE(param.maximumRadius() == static_cast<Scalar>(0.78));
                REQUIRE(param.radiusEpsilon() == static_cast<Scalar>(1e-2));
                REQUIRE(param.acceptanceFitness() == static_cast<Scalar>(0.11));
                REQUIRE(param.maximumIterations() == 10);
            }
        }

        SECTION("setter")
        {
            SECTION("maximum radius")
            {
                Parameter param;
                REQUIRE(param.initialRadius() == static_cast<Scalar>(1));
                REQUIRE(param.maximumRadius() == static_cast<Scalar>(2));

                param.setMaximumRadius(static_cast<Scalar>(0.78));
                REQUIRE(param.initialRadius() == static_cast<Scalar>(0.78));
                REQUIRE(param.maximumRadius() == static_cast<Scalar>(0.78));
            }

            SECTION("radius epsilon")
            {
                Parameter param;
                REQUIRE(param.radiusEpsilon() == static_cast<Scalar>(1e-6));

                param.setRadiusEpsilon(static_cast<Scalar>(1e-2));
                REQUIRE(param.radiusEpsilon() == static_cast<Scalar>(1e-2));
            }

            SECTION("acceptance fitness")
            {
                Parameter param;
                REQUIRE(param.acceptanceFitness() == static_cast<Scalar>(0.25));

                param.setAcceptanceFitness(static_cast<Scalar>(0.11));
                REQUIRE(param.acceptanceFitness() == static_cast<Scalar>(0.11));
            }

            SECTION("maximum iterations")
            {
                Parameter param;
                REQUIRE(param.maximumIterations() == 0);

                param.setMaximumIterations(10);
                REQUIRE(param.maximumIterations() == 10);
            }
        }
    }

    SECTION("refinement")
    {
        SECTION("dynamic size problem")
        {
            constexpr int Inputs = Eigen::Dynamic;
            constexpr int Outputs = Eigen::Dynamic;
            using InputVector = Eigen::Matrix<Scalar, Inputs, 1>;
            using OutputVector = Eigen::Matrix<Scalar, Outputs, 1>;
            using JacobiMatrix = Eigen::Matrix<Scalar, Outputs, Inputs>;
            using GradientVector = Eigen::Matrix<Scalar, Inputs, 1>;
            using StepVector = Eigen::Matrix<Scalar, Inputs, 1>;
            using Refiner = NewtonStepRefiner<Scalar, Inputs, Outputs, DoglegMethod>;
            constexpr auto eps = static_cast<Scalar>(1e-6);

            ParabolicError objective;
            InputVector xval(4);
            xval << 1, 2, 1, 2;
            OutputVector fval;
            JacobiMatrix jacobian;
            objective(xval, fval, jacobian);

            GradientVector gradient = jacobian.transpose() * fval;
            StepVector step = gradient;

            Parameter param;
            Refiner refiner(param);
            StepVector expected(4);
            expected << static_cast<Scalar>(0.316228),
                        static_cast<Scalar>(0.632456),
                        static_cast<Scalar>(0.316228),
                        static_cast<Scalar>(0.632456);

            refiner(xval, fval, jacobian, gradient, objective, step);

            REQUIRE_MATRIX_APPROX(expected, step, eps);
        }

        SECTION("fixed size problem")
        {
            constexpr int Inputs = 4;
            constexpr int Outputs = 2;
            using InputVector = Eigen::Matrix<Scalar, Inputs, 1>;
            using OutputVector = Eigen::Matrix<Scalar, Outputs, 1>;
            using JacobiMatrix = Eigen::Matrix<Scalar, Outputs, Inputs>;
            using GradientVector = Eigen::Matrix<Scalar, Inputs, 1>;
            using StepVector = Eigen::Matrix<Scalar, Inputs, 1>;
            using Refiner = NewtonStepRefiner<Scalar, Inputs, Outputs, DoglegMethod>;
            constexpr auto eps = static_cast<Scalar>(1e-6);

            ParabolicError objective;
            InputVector xval(4);
            xval << 1, 2, 1, 2;
            OutputVector fval;
            JacobiMatrix jacobian;
            objective(xval, fval, jacobian);

            GradientVector gradient = jacobian.transpose() * fval;
            StepVector step = gradient;

            Parameter param;
            Refiner refiner(param);
            StepVector expected(4);
            expected << static_cast<Scalar>(0.316228),
                        static_cast<Scalar>(0.632456),
                        static_cast<Scalar>(0.316228),
                        static_cast<Scalar>(0.632456);


            refiner(xval, fval, jacobian, gradient, objective, step);

            REQUIRE_MATRIX_APPROX(expected, step, eps);
        }
    }

}