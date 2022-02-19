/// step_refiner_dogleg_method.test.cpp
///
/// Author:     Fabian Meyer
/// Created On: 22 Jan 2021
/// License:    MIT

#include <lsqcpp/lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsq;

TEMPLATE_TEST_CASE("dogleg method step refiner ", "[step refiner]", float, double)
{
    using Scalar = TestType;
    using DynamicRefiner = NewtonStepRefiner<Scalar, Eigen::Dynamic, Eigen::Dynamic, DoglegMethod>;

    SECTION("construction")
    {
        SECTION("default")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.radius() == static_cast<Scalar>(1));
            REQUIRE(refiner.maximumRadius() == static_cast<Scalar>(2));
            REQUIRE(refiner.radiusEpsilon() == static_cast<Scalar>(1e-6));
            REQUIRE(refiner.acceptanceFitness() == static_cast<Scalar>(0.25));
            REQUIRE(refiner.maximumIterations() == 0);
        }

        SECTION("parametrized A")
        {
            DynamicRefiner refiner(static_cast<Scalar>(0.42),
                                   static_cast<Scalar>(0.78),
                                   static_cast<Scalar>(1e-2),
                                   static_cast<Scalar>(0.11),
                                   10);
            REQUIRE(refiner.radius() == static_cast<Scalar>(0.42));
            REQUIRE(refiner.maximumRadius() == static_cast<Scalar>(0.78));
            REQUIRE(refiner.radiusEpsilon() == static_cast<Scalar>(1e-2));
            REQUIRE(refiner.acceptanceFitness() == static_cast<Scalar>(0.11));
            REQUIRE(refiner.maximumIterations() == 10);
        }
    }

    SECTION("setter")
    {
        SECTION("maximum radius")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.radius() == static_cast<Scalar>(1));
            REQUIRE(refiner.maximumRadius() == static_cast<Scalar>(2));

            refiner.setMaximumRadius(static_cast<Scalar>(0.78));
            REQUIRE(refiner.radius() == static_cast<Scalar>(0.78));
            REQUIRE(refiner.maximumRadius() == static_cast<Scalar>(0.78));
        }

        SECTION("radius epsilon")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.radiusEpsilon() == static_cast<Scalar>(1e-6));

            refiner.setRadiusEpsilon(static_cast<Scalar>(1e-2));
            REQUIRE(refiner.radiusEpsilon() == static_cast<Scalar>(1e-2));
        }

        SECTION("acceptance fitness")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.acceptanceFitness() == static_cast<Scalar>(0.25));

            refiner.setAcceptanceFitness(static_cast<Scalar>(0.11));
            REQUIRE(refiner.acceptanceFitness() == static_cast<Scalar>(0.11));
        }

        SECTION("maximum iterations")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.maximumIterations() == 0);

            refiner.setMaximumIterations(10);
            REQUIRE(refiner.maximumIterations() == 10);
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

            Refiner refiner;
            StepVector expected(4);
            expected << static_cast<Scalar>(-0.316228),
                        static_cast<Scalar>(-0.632456),
                        static_cast<Scalar>(-0.316228),
                        static_cast<Scalar>(-0.632456);

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

            Refiner refiner;
            StepVector expected(4);
            expected << static_cast<Scalar>(-0.316228),
                        static_cast<Scalar>(-0.632456),
                        static_cast<Scalar>(-0.316228),
                        static_cast<Scalar>(-0.632456);


            refiner(xval, fval, jacobian, gradient, objective, step);

            REQUIRE_MATRIX_APPROX(expected, step, eps);
        }
    }

}