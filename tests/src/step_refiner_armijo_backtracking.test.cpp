/// step_refiner_armijo_backtracking.test.cpp
///
/// Author:     Fabian Meyer
/// Created On: 22 Jan 2021
/// License:    MIT

#include <lsqcpp/lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsqcpp;

TEMPLATE_TEST_CASE("armijo backtracking step refiner", "[step refiner]", float, double)
{
    using Scalar = TestType;
    using Parameter = ArmijoBacktrackingParameter<Scalar>;

    SECTION("parameter")
    {
        SECTION("construction")
        {
            SECTION("default")
            {
                Parameter param;
                REQUIRE(param.backtrackingDecrease() == static_cast<Scalar>(0.8));
                REQUIRE(param.armijoConstant() == static_cast<Scalar>(1e-4));
                REQUIRE(param.minimumStepBound() == static_cast<Scalar>(1e-10));
                REQUIRE(param.maximumStepBound() == static_cast<Scalar>(1));
                REQUIRE(param.maximumIterations() == 0);
            }

            SECTION("parametrized A")
            {
                Parameter param(static_cast<Scalar>(0.42),
                                       static_cast<Scalar>(1e-2),
                                       static_cast<Scalar>(1e-4),
                                       static_cast<Scalar>(1e-3),
                                       10);
                REQUIRE(param.backtrackingDecrease() == static_cast<Scalar>(0.42));
                REQUIRE(param.armijoConstant() == static_cast<Scalar>(1e-2));
                REQUIRE(param.minimumStepBound() == static_cast<Scalar>(1e-4));
                REQUIRE(param.maximumStepBound() == static_cast<Scalar>(1e-3));
                REQUIRE(param.maximumIterations() == 10);
            }
        }

        SECTION("setter")
        {
            SECTION("backtracking decrease")
            {
                Parameter param;
                REQUIRE(param.backtrackingDecrease() == static_cast<Scalar>(0.8));

                param.setBacktrackingDecrease(static_cast<Scalar>(0.42));
                REQUIRE(param.backtrackingDecrease() == static_cast<Scalar>(0.42));
            }

            SECTION("armijo constant")
            {
                Parameter param;
                REQUIRE(param.armijoConstant() == static_cast<Scalar>(1e-4));

                param.setArmijoConstant(static_cast<Scalar>(1e-2));
                REQUIRE(param.armijoConstant() == static_cast<Scalar>(1e-2));
            }

            SECTION("step bounds")
            {
                Parameter param;
                REQUIRE(param.minimumStepBound() == static_cast<Scalar>(1e-10));
                REQUIRE(param.maximumStepBound() == static_cast<Scalar>(1));

                param.setStepBounds(static_cast<Scalar>(1e-4), static_cast<Scalar>(1e-3));
                REQUIRE(param.minimumStepBound() == static_cast<Scalar>(1e-4));
                REQUIRE(param.maximumStepBound() == static_cast<Scalar>(1e-3));
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
            using Refiner = NewtonStepRefiner<Scalar, Inputs, Outputs, ArmijoBacktracking>;
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
            expected << static_cast<Scalar>(1.67772),
                        static_cast<Scalar>(3.35544),
                        static_cast<Scalar>(1.67772),
                        static_cast<Scalar>(3.35544);

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
            using Refiner = NewtonStepRefiner<Scalar, Inputs, Outputs, ArmijoBacktracking>;
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
            expected << static_cast<Scalar>(1.67772),
                        static_cast<Scalar>(3.35544),
                        static_cast<Scalar>(1.67772),
                        static_cast<Scalar>(3.35544);

            refiner(xval, fval, jacobian, gradient, objective, step);

            REQUIRE_MATRIX_APPROX(expected, step, eps);
        }
    }

}