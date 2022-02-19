/// step_refiner_levenberg_marquardt.test.cpp
///
/// Author:     Fabian Meyer
/// Created On: 04 Feb 2022
/// License:    MIT

#include <lsqcpp/lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsqcpp;

TEMPLATE_TEST_CASE("levenberg marquardt step refiner", "[step refiner]", float, double)
{
    using Scalar = TestType;
    using DynamicRefiner = NewtonStepRefiner<Scalar, Eigen::Dynamic, Eigen::Dynamic, LevenbergMarquardtMethod>;

    SECTION("construction")
    {
        SECTION("default")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.lambda() == static_cast<Scalar>(1));
            REQUIRE(refiner.increase() == static_cast<Scalar>(2));
            REQUIRE(refiner.decrease() == static_cast<Scalar>(0.5));
            REQUIRE(refiner.maximumIterations() == 0);
        }

        SECTION("parametrized A")
        {
            DynamicRefiner refiner(static_cast<Scalar>(0.42),
                                   static_cast<Scalar>(2.25),
                                   static_cast<Scalar>(0.72),
                                   10,
                                   DenseSVDSolver());
            REQUIRE(refiner.lambda() == static_cast<Scalar>(0.42));
            REQUIRE(refiner.increase() == static_cast<Scalar>(2.25));
            REQUIRE(refiner.decrease() == static_cast<Scalar>(0.72));
            REQUIRE(refiner.maximumIterations() == 10);
        }
    }

    SECTION("setter")
    {
        SECTION("lambda")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.lambda() == static_cast<Scalar>(1));

            refiner.setLambda(static_cast<Scalar>(0.42));
            REQUIRE(refiner.lambda() == static_cast<Scalar>(0.42));
        }

        SECTION("increase")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.increase() == static_cast<Scalar>(2));

            refiner.setIncrease(static_cast<Scalar>(2.25));
            REQUIRE(refiner.increase() == static_cast<Scalar>(2.25));
        }

        SECTION("decrease")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.decrease() == static_cast<Scalar>(0.5));

            refiner.setDecrease(static_cast<Scalar>(0.72));
            REQUIRE(refiner.decrease() == static_cast<Scalar>(0.72));
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
            using Refiner = NewtonStepRefiner<Scalar, Inputs, Outputs, LevenbergMarquardtMethod>;
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
            expected << static_cast<Scalar>(0.4761904),
                        static_cast<Scalar>(0.9523809),
                        static_cast<Scalar>(0.4761904),
                        static_cast<Scalar>(0.9523809);

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
            using Refiner = NewtonStepRefiner<Scalar, Inputs, Outputs, LevenbergMarquardtMethod>;
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
            refiner.setSolver(DenseSVDSolver());
            StepVector expected(4);
            expected << static_cast<Scalar>(0.4761904),
                        static_cast<Scalar>(0.9523809),
                        static_cast<Scalar>(0.4761904),
                        static_cast<Scalar>(0.9523809);

            refiner(xval, fval, jacobian, gradient, objective, step);

            REQUIRE_MATRIX_APPROX(expected, step, eps);
        }
    }

}