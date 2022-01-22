/* step_refiner_armijo_backtracking_test.cpp
 *
 * Author: Fabian Meyer
 * Created On: 22 Jan 2021
 */

#include <lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsq;

TEMPLATE_TEST_CASE("armijo backtracking step refiner ", "[step refiner]", float, double)
{
    using Scalar = TestType;
    using DynamicRefiner = NewtonStepRefiner<Scalar, Eigen::Dynamic, Eigen::Dynamic, ArmijoBacktracking>;

    SECTION("construction")
    {
        SECTION("default")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.backtrackingDecrease() == static_cast<Scalar>(0.8));
            REQUIRE(refiner.armijoConstant() == static_cast<Scalar>(1e-4));
            REQUIRE(refiner.minimumStepBound() == static_cast<Scalar>(1e-10));
            REQUIRE(refiner.maximumStepBound() == static_cast<Scalar>(1));
            REQUIRE(refiner.maximumIterations() == 0);
        }

        SECTION("parametrized A")
        {
            DynamicRefiner refiner(static_cast<Scalar>(0.42),
                                   static_cast<Scalar>(1e-2),
                                   static_cast<Scalar>(1e-4),
                                   static_cast<Scalar>(1e-3),
                                   10);
            REQUIRE(refiner.backtrackingDecrease() == static_cast<Scalar>(0.42));
            REQUIRE(refiner.armijoConstant() == static_cast<Scalar>(1e-2));
            REQUIRE(refiner.minimumStepBound() == static_cast<Scalar>(1e-4));
            REQUIRE(refiner.maximumStepBound() == static_cast<Scalar>(1e-3));
            REQUIRE(refiner.maximumIterations() == 10);
        }
    }

    SECTION("setter")
    {
        SECTION("backtracking decrease")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.backtrackingDecrease() == static_cast<Scalar>(0.8));

            refiner.setBacktrackingDecrease(static_cast<Scalar>(0.42));
            REQUIRE(refiner.backtrackingDecrease() == static_cast<Scalar>(0.42));
        }

        SECTION("armijo constant")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.armijoConstant() == static_cast<Scalar>(1e-4));

            refiner.setArmijoConstant(static_cast<Scalar>(1e-2));
            REQUIRE(refiner.armijoConstant() == static_cast<Scalar>(1e-2));
        }

        SECTION("step bounds")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.minimumStepBound() == static_cast<Scalar>(1e-10));
            REQUIRE(refiner.maximumStepBound() == static_cast<Scalar>(1));

            refiner.setStepBounds(static_cast<Scalar>(1e-4), static_cast<Scalar>(1e-3));
            REQUIRE(refiner.minimumStepBound() == static_cast<Scalar>(1e-4));
            REQUIRE(refiner.maximumStepBound() == static_cast<Scalar>(1e-3));
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
            // constexpr int Inputs = Eigen::Dynamic;
            // constexpr int Outputs = Eigen::Dynamic;
            // using InputVector = Eigen::Matrix<Scalar, Inputs, 1>;
            // using OutputVector = Eigen::Matrix<Scalar, Outputs, 1>;
            // using JacobiMatrix = Eigen::Matrix<Scalar, Outputs, Inputs>;
            // using GradientVector = Eigen::Matrix<Scalar, Inputs, 1>;
            // using StepVector = Eigen::Matrix<Scalar, Inputs, 1>;
            // using Refiner = NewtonStepRefiner<Scalar, Inputs, Outputs, BarzilaiBorwein>;
            // constexpr auto eps = static_cast<Scalar>(1e-6);

            // InputVector xval(4);
            // xval << 1, 2, 1, 2;
            // OutputVector fval;
            // JacobiMatrix jacobian;
            // GradientVector gradient;
            // StepVector step(4);
            // ParabolicError objective;
            // step << static_cast<Scalar>(1),
            //         static_cast<Scalar>(2.5),
            //         static_cast<Scalar>(-5.25),
            //         static_cast<Scalar>(-2.5);

            // SECTION("direct")
            // {
            //     Refiner refiner(BarzilaiBorwein::Mode::Direct, static_cast<Scalar>(2.5));
            //     StepVector expected = step * static_cast<Scalar>(2.5) / step.norm();

            //     refiner(xval, fval, jacobian, gradient, objective, step);
            //     REQUIRE_MATRIX_APPROX(expected, step, eps);

            //     xval << 3, 1, 2, 2;
            //     step << static_cast<Scalar>(0.5),
            //             static_cast<Scalar>(3.5),
            //             static_cast<Scalar>(-4.25),
            //             static_cast<Scalar>(-1.5);

            //     expected << -3, -21, static_cast<Scalar>(25.5), 9;
            //     refiner(xval, fval, jacobian, gradient, objective, step);
            //     REQUIRE_MATRIX_APPROX(expected, step, eps);
            // }

            // SECTION("inverse")
            // {
            //     Refiner refiner(BarzilaiBorwein::Mode::Inverse, static_cast<Scalar>(2.5));
            //     StepVector expected = step * static_cast<Scalar>(2.5) / step.norm();

            //     refiner(xval, fval, jacobian, gradient, objective, step);
            //     REQUIRE_MATRIX_APPROX(expected, step, eps);

            //     xval << 3, 1, 2, 2;
            //     step << static_cast<Scalar>(2),
            //             static_cast<Scalar>(3.5),
            //             static_cast<Scalar>(-4.25),
            //             static_cast<Scalar>(-1.5);

            //     expected << 1, static_cast<Scalar>(1.75), static_cast<Scalar>(-2.125), static_cast<Scalar>(-0.75);
            //     refiner(xval, fval, jacobian, gradient, objective, step);
            //     REQUIRE_MATRIX_APPROX(expected, step, eps);
            // }
        }

        SECTION("fixed size problem")
        {
            // constexpr int Inputs = 4;
            // constexpr int Outputs = 2;
            // using InputVector = Eigen::Matrix<Scalar, Inputs, 1>;
            // using OutputVector = Eigen::Matrix<Scalar, Outputs, 1>;
            // using JacobiMatrix = Eigen::Matrix<Scalar, Outputs, Inputs>;
            // using GradientVector = Eigen::Matrix<Scalar, Inputs, 1>;
            // using StepVector = Eigen::Matrix<Scalar, Inputs, 1>;
            // using Refiner = NewtonStepRefiner<Scalar, Inputs, Outputs, BarzilaiBorwein>;
            // constexpr auto eps = static_cast<Scalar>(1e-6);

            // InputVector xval;
            // xval << 1, 2, 1, 2;
            // OutputVector fval;
            // JacobiMatrix jacobian;
            // GradientVector gradient;
            // StepVector step;
            // ParabolicError objective;
            // step << static_cast<Scalar>(1),
            //         static_cast<Scalar>(2.5),
            //         static_cast<Scalar>(-5.25),
            //         static_cast<Scalar>(-2.5);

            // SECTION("direct")
            // {
            //     Refiner refiner(BarzilaiBorwein::Mode::Direct, static_cast<Scalar>(2.5));
            //     StepVector expected = step * static_cast<Scalar>(2.5) / step.norm();

            //     refiner(xval, fval, jacobian, gradient, objective, step);
            //     REQUIRE_MATRIX_APPROX(expected, step, eps);

            //     xval << 3, 1, 2, 2;
            //     step << static_cast<Scalar>(0.5),
            //             static_cast<Scalar>(3.5),
            //             static_cast<Scalar>(-4.25),
            //             static_cast<Scalar>(-1.5);

            //     expected << -3, -21, static_cast<Scalar>(25.5), 9;
            //     refiner(xval, fval, jacobian, gradient, objective, step);
            //     REQUIRE_MATRIX_APPROX(expected, step, eps);
            // }

            // SECTION("inverse")
            // {
            //     Refiner refiner(BarzilaiBorwein::Mode::Inverse, static_cast<Scalar>(2.5));
            //     StepVector expected = step * static_cast<Scalar>(2.5) / step.norm();

            //     refiner(xval, fval, jacobian, gradient, objective, step);
            //     REQUIRE_MATRIX_APPROX(expected, step, eps);

            //     xval << 3, 1, 2, 2;
            //     step << static_cast<Scalar>(2),
            //             static_cast<Scalar>(3.5),
            //             static_cast<Scalar>(-4.25),
            //             static_cast<Scalar>(-1.5);

            //     expected << 1, static_cast<Scalar>(1.75), static_cast<Scalar>(-2.125), static_cast<Scalar>(-0.75);
            //     refiner(xval, fval, jacobian, gradient, objective, step);
            //     REQUIRE_MATRIX_APPROX(expected, step, eps);
            // }
        }
    }

}