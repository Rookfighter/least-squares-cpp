/// step_refiner_wolfe_backtracking.test.cpp
///
/// Author:     Fabian Meyer
/// Created On: 22 Jan 2021
/// License:    MIT


#include <lsqcpp/lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsq;

TEMPLATE_TEST_CASE("wolfe backtracking step refiner", "[step refiner]", float, double)
{
    using Scalar = TestType;
    using DynamicRefiner = NewtonStepRefiner<Scalar, Eigen::Dynamic, Eigen::Dynamic, WolfeBacktracking>;

    SECTION("construction")
    {
        SECTION("default")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.backtrackingDecrease() == static_cast<Scalar>(0.8));
            REQUIRE(refiner.armijoConstant() == static_cast<Scalar>(1e-4));
            REQUIRE(refiner.wolfeConstant() == static_cast<Scalar>(0.9));
            REQUIRE(refiner.minimumStepBound() == static_cast<Scalar>(1e-10));
            REQUIRE(refiner.maximumStepBound() == static_cast<Scalar>(1));
            REQUIRE(refiner.maximumIterations() == 0);
        }

        SECTION("parametrized A")
        {
            DynamicRefiner refiner(static_cast<Scalar>(0.42),
                                   static_cast<Scalar>(1e-2),
                                   static_cast<Scalar>(0.72),
                                   static_cast<Scalar>(1e-4),
                                   static_cast<Scalar>(1e-3),
                                   10);
            REQUIRE(refiner.backtrackingDecrease() == static_cast<Scalar>(0.42));
            REQUIRE(refiner.armijoConstant() == static_cast<Scalar>(1e-2));
            REQUIRE(refiner.wolfeConstant() == static_cast<Scalar>(0.72));
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

        SECTION("wolfe constants")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.armijoConstant() == static_cast<Scalar>(1e-4));
            REQUIRE(refiner.wolfeConstant() == static_cast<Scalar>(0.9));

            refiner.setWolfeConstants(static_cast<Scalar>(1e-2), static_cast<Scalar>(0.72));
            REQUIRE(refiner.armijoConstant() == static_cast<Scalar>(1e-2));
            REQUIRE(refiner.wolfeConstant() == static_cast<Scalar>(0.72));
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
            constexpr int Inputs = Eigen::Dynamic;
            constexpr int Outputs = Eigen::Dynamic;
            using InputVector = Eigen::Matrix<Scalar, Inputs, 1>;
            using OutputVector = Eigen::Matrix<Scalar, Outputs, 1>;
            using JacobiMatrix = Eigen::Matrix<Scalar, Outputs, Inputs>;
            using GradientVector = Eigen::Matrix<Scalar, Inputs, 1>;
            using StepVector = Eigen::Matrix<Scalar, Inputs, 1>;
            using Refiner = NewtonStepRefiner<Scalar, Inputs, Outputs, WolfeBacktracking>;
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
            expected << static_cast<Scalar>(0.0302232),
                        static_cast<Scalar>(0.0604463),
                        static_cast<Scalar>(0.0302232),
                        static_cast<Scalar>(0.0604463);

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
            using Refiner = NewtonStepRefiner<Scalar, Inputs, Outputs, WolfeBacktracking>;
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
            expected << static_cast<Scalar>(0.0302232),
                        static_cast<Scalar>(0.0604463),
                        static_cast<Scalar>(0.0302232),
                        static_cast<Scalar>(0.0604463);

            refiner(xval, fval, jacobian, gradient, objective, step);

            REQUIRE_MATRIX_APPROX(expected, step, eps);
        }
    }

}