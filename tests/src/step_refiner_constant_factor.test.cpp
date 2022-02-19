/// step_refiner_constant_factor.test.cpp
///
/// Author:     Fabian Meyer
/// Created On: 22 Jan 2021
/// License:    MIT

#include <lsqcpp/lsqcpp.h>
#include "eigen_require.h"
#include "parabolic_error.h"

using namespace lsq;

TEMPLATE_TEST_CASE("constant factor step refiner", "[step refiner]", float, double)
{
    using Scalar = TestType;
    using DynamicRefiner = NewtonStepRefiner<Scalar, Eigen::Dynamic, Eigen::Dynamic, ConstantStepFactor>;

    SECTION("construction")
    {
        SECTION("default")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.factor() == static_cast<Scalar>(1));
        }

        SECTION("parametrized A")
        {
            DynamicRefiner refiner(14);
            REQUIRE(refiner.factor() == static_cast<Scalar>(14));
        }
    }

    SECTION("setter")
    {
        SECTION("factor")
        {
            DynamicRefiner refiner;
            REQUIRE(refiner.factor() == static_cast<Scalar>(1));

            refiner.setFactor(7);
            REQUIRE(refiner.factor() == static_cast<Scalar>(7));
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
            using Refiner = NewtonStepRefiner<Scalar, Inputs, Outputs, ConstantStepFactor>;
            constexpr auto eps = static_cast<Scalar>(1e-6);

            InputVector xval;
            OutputVector fval;
            JacobiMatrix jacobian;
            GradientVector gradient;
            StepVector step(4);
            ParabolicError objective;
            step << static_cast<Scalar>(1),
                    static_cast<Scalar>(2.5),
                    static_cast<Scalar>(-5.25),
                    static_cast<Scalar>(21.4);

            Refiner refiner(static_cast<Scalar>(2.5));
            StepVector expected = step * static_cast<Scalar>(2.5);

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
            using Refiner = NewtonStepRefiner<Scalar, Inputs, Outputs, ConstantStepFactor>;
            constexpr auto eps = static_cast<Scalar>(1e-6);

            InputVector xval;
            OutputVector fval;
            JacobiMatrix jacobian;
            GradientVector gradient;
            StepVector step(4);
            ParabolicError objective;
            step << static_cast<Scalar>(1),
                    static_cast<Scalar>(2.5),
                    static_cast<Scalar>(-5.25),
                    static_cast<Scalar>(21.4);

            Refiner refiner(static_cast<Scalar>(2.5));
            StepVector expected = step * static_cast<Scalar>(2.5);

            refiner(xval, fval, jacobian, gradient, objective, step);

            REQUIRE_MATRIX_APPROX(expected, step, eps);
        }
    }

}