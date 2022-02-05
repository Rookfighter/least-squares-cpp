/// step_refiner_barzilai_borwein_test.cpp
///
/// Author: Fabian Meyer
/// Created On: 22 Jan 2021


#include <lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsq;

TEMPLATE_TEST_CASE("barzilai borwein step refiner", "[step refiner]", float, double)
{
    using Scalar = TestType;
    using DynamicRefiner = NewtonStepRefiner<Scalar, Eigen::Dynamic, Eigen::Dynamic, BarzilaiBorwein>;

    SECTION("construction")
    {
        SECTION("default")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.mode() == BarzilaiBorwein::Mode::Direct);
            REQUIRE(refiner.constantStepSize() == static_cast<Scalar>(1e-2));
        }

        SECTION("parametrized A")
        {
            DynamicRefiner refiner(BarzilaiBorwein::Mode::Inverse);
            REQUIRE(refiner.mode() == BarzilaiBorwein::Mode::Inverse);
            REQUIRE(refiner.constantStepSize() == static_cast<Scalar>(1e-2));
        }

        SECTION("parametrized B")
        {
            DynamicRefiner refiner(3);
            REQUIRE(refiner.mode() == BarzilaiBorwein::Mode::Direct);
            REQUIRE(refiner.constantStepSize() == static_cast<Scalar>(3));
        }

        SECTION("parametrized C")
        {
            DynamicRefiner refiner(BarzilaiBorwein::Mode::Inverse, 14);
            REQUIRE(refiner.mode() == BarzilaiBorwein::Mode::Inverse);
            REQUIRE(refiner.constantStepSize() == static_cast<Scalar>(14));
        }
    }

    SECTION("setter")
    {
        SECTION("mode")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.mode() == BarzilaiBorwein::Mode::Direct);

            refiner.setMode(BarzilaiBorwein::Mode::Inverse);
            REQUIRE(refiner.mode() == BarzilaiBorwein::Mode::Inverse);
        }

        SECTION("constant step size")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.constantStepSize() == static_cast<Scalar>(1e-2));

            refiner.setConstantStepSize(5);
            REQUIRE(refiner.constantStepSize() == static_cast<Scalar>(5));
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
            using Refiner = NewtonStepRefiner<Scalar, Inputs, Outputs, BarzilaiBorwein>;
            constexpr auto eps = static_cast<Scalar>(1e-6);

            InputVector xval(4);
            xval << 1, 2, 1, 2;
            OutputVector fval;
            JacobiMatrix jacobian;
            GradientVector gradient;
            StepVector step(4);
            ParabolicError objective;
            step << static_cast<Scalar>(1),
                    static_cast<Scalar>(2.5),
                    static_cast<Scalar>(-5.25),
                    static_cast<Scalar>(-2.5);

            SECTION("direct")
            {
                Refiner refiner(BarzilaiBorwein::Mode::Direct, static_cast<Scalar>(2.5));
                StepVector expected = step/// static_cast<Scalar>(2.5) / step.norm();

                refiner(xval, fval, jacobian, gradient, objective, step);
                REQUIRE_MATRIX_APPROX(expected, step, eps);

                xval << 3, 1, 2, 2;
                step << static_cast<Scalar>(0.5),
                        static_cast<Scalar>(3.5),
                        static_cast<Scalar>(-4.25),
                        static_cast<Scalar>(-1.5);

                expected << -3, -21, static_cast<Scalar>(25.5), 9;
                refiner(xval, fval, jacobian, gradient, objective, step);
                REQUIRE_MATRIX_APPROX(expected, step, eps);
            }

            SECTION("inverse")
            {
                Refiner refiner(BarzilaiBorwein::Mode::Inverse, static_cast<Scalar>(2.5));
                StepVector expected = step/// static_cast<Scalar>(2.5) / step.norm();

                refiner(xval, fval, jacobian, gradient, objective, step);
                REQUIRE_MATRIX_APPROX(expected, step, eps);

                xval << 3, 1, 2, 2;
                step << static_cast<Scalar>(2),
                        static_cast<Scalar>(3.5),
                        static_cast<Scalar>(-4.25),
                        static_cast<Scalar>(-1.5);

                expected << 1, static_cast<Scalar>(1.75), static_cast<Scalar>(-2.125), static_cast<Scalar>(-0.75);
                refiner(xval, fval, jacobian, gradient, objective, step);
                REQUIRE_MATRIX_APPROX(expected, step, eps);
            }
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
            using Refiner = NewtonStepRefiner<Scalar, Inputs, Outputs, BarzilaiBorwein>;
            constexpr auto eps = static_cast<Scalar>(1e-6);

            InputVector xval;
            xval << 1, 2, 1, 2;
            OutputVector fval;
            JacobiMatrix jacobian;
            GradientVector gradient;
            StepVector step;
            ParabolicError objective;
            step << static_cast<Scalar>(1),
                    static_cast<Scalar>(2.5),
                    static_cast<Scalar>(-5.25),
                    static_cast<Scalar>(-2.5);

            SECTION("direct")
            {
                Refiner refiner(BarzilaiBorwein::Mode::Direct, static_cast<Scalar>(2.5));
                StepVector expected = step/// static_cast<Scalar>(2.5) / step.norm();

                refiner(xval, fval, jacobian, gradient, objective, step);
                REQUIRE_MATRIX_APPROX(expected, step, eps);

                xval << 3, 1, 2, 2;
                step << static_cast<Scalar>(0.5),
                        static_cast<Scalar>(3.5),
                        static_cast<Scalar>(-4.25),
                        static_cast<Scalar>(-1.5);

                expected << -3, -21, static_cast<Scalar>(25.5), 9;
                refiner(xval, fval, jacobian, gradient, objective, step);
                REQUIRE_MATRIX_APPROX(expected, step, eps);
            }

            SECTION("inverse")
            {
                Refiner refiner(BarzilaiBorwein::Mode::Inverse, static_cast<Scalar>(2.5));
                StepVector expected = step/// static_cast<Scalar>(2.5) / step.norm();

                refiner(xval, fval, jacobian, gradient, objective, step);
                REQUIRE_MATRIX_APPROX(expected, step, eps);

                xval << 3, 1, 2, 2;
                step << static_cast<Scalar>(2),
                        static_cast<Scalar>(3.5),
                        static_cast<Scalar>(-4.25),
                        static_cast<Scalar>(-1.5);

                expected << 1, static_cast<Scalar>(1.75), static_cast<Scalar>(-2.125), static_cast<Scalar>(-0.75);
                refiner(xval, fval, jacobian, gradient, objective, step);
                REQUIRE_MATRIX_APPROX(expected, step, eps);
            }
        }
    }

}