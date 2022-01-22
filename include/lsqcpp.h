/* lsqcpp.h
 *
 * Author: Fabian Meyer
 * Created On: 22 Jul 2019
 * License: MIT
 */

#ifndef LSQCPP_LSQCPP_H_
#define LSQCPP_LSQCPP_H_

#include <Eigen/Geometry>
#include <vector>
#include <limits>
#include <iostream>
#include <iomanip>
#include <functional>
#include <type_traits>

namespace lsq
{
    using Index = Eigen::MatrixXd::Index;

    /** Functor to compute forward differences.
      * Computes the gradient of the objective f(x) as follows:
      *
      * grad(x) = (f(x + eps) - f(x)) / eps
      *
      * The computation requires len(x) evaluations of the objective. */
    struct ForwardDifferences
    {
        template<typename Scalar, int Inputs, int Outputs, typename Objective>
        void operator()(const Eigen::Matrix<Scalar, Inputs, 1> &xval,
                          const Eigen::Matrix<Scalar, Outputs, 1> &fval,
                          const Objective &objective,
                          const Scalar eps,
                          const int threads,
                          Eigen::Matrix<Scalar, Outputs, Inputs> &jacobian) const
        {
            static_assert(Eigen::NumTraits<Scalar>::IsInteger == 0, "Finite differences only supports non-integer scalars.");

            using InputVector = Eigen::Matrix<Scalar, Inputs, 1>;
            using OutputVector = Eigen::Matrix<Scalar, Outputs, 1>;

            // noop in fixed size problems
            jacobian.resize(fval.size(), xval.size());

            #pragma omp parallel for num_threads(threads)
            for(Index i = 0; i < xval.size(); ++i)
            {
                InputVector xvalNext = xval;
                OutputVector fvalNext;
                xvalNext(i) += eps;
                objective(xvalNext, fvalNext);

                jacobian.col(i) = (fvalNext - fval) / eps;
            }
        }
    };

    /** Functor to compute backward differences.
      * Computes the gradient of the objective f(x) as follows:
      *
      * grad(x) = (f(x) - f(x - eps)) / eps
      *
      * The computation requires len(x) evaluations of the objective. */
    struct BackwardDifferences
    {
        template<typename Scalar, int Inputs, int Outputs, typename Objective>
        void operator()(const Eigen::Matrix<Scalar, Inputs, 1> &xval,
                          const Eigen::Matrix<Scalar, Outputs, 1> &fval,
                          const Objective &objective,
                          const Scalar eps,
                          const int threads,
                          Eigen::Matrix<Scalar, Outputs, Inputs> &jacobian) const
        {
            static_assert(Eigen::NumTraits<Scalar>::IsInteger == 0, "Finite differences only supports non-integer scalars.");

            using InputVector = Eigen::Matrix<Scalar, Inputs, 1>;
            using OutputVector = Eigen::Matrix<Scalar, Outputs, 1>;

            // noop in fixed size problems
            jacobian.resize(fval.size(), xval.size());

            #pragma omp parallel for num_threads(threads)
            for(Index i = 0; i < xval.size(); ++i)
            {
                InputVector xvalNext = xval;
                OutputVector fvalNext;
                xvalNext(i) -= eps;
                objective(xvalNext, fvalNext);

                jacobian.col(i) = (fval - fvalNext) / eps;
            }
        }
    };

    /** Functor to compute central differences.
      * Computes the gradient of the objective f(x) as follows:
      *
      * grad(x) = (f(x + 0.5 eps) - f(x - 0.5 eps)) / eps
      *
      * The computation requires 2 * len(x) evaluations of the objective. */
    struct CentralDifferences
    {
        template<typename Scalar, int Inputs, int Outputs, typename Objective>
        void operator()(const Eigen::Matrix<Scalar, Inputs, 1> &xval,
                          const Eigen::Matrix<Scalar, Outputs, 1> &fval,
                          const Objective &objective,
                          const Scalar eps,
                          const int threads,
                          Eigen::Matrix<Scalar, Outputs, Inputs> &jacobian) const
        {
            static_assert(Eigen::NumTraits<Scalar>::IsInteger == 0, "Finite differences only supports non-integer scalars.");

            using InputVector = Eigen::Matrix<Scalar, Inputs, 1>;
            using OutputVector = Eigen::Matrix<Scalar, Outputs, 1>;

            // noop in fixed size problems
            jacobian.resize(fval.size(), xval.size());

            #pragma omp parallel for num_threads(threads)
            for(Index i = 0; i < xval.size(); ++i)
            {
                InputVector xvalNext = xval;
                OutputVector fvalA;
                OutputVector fvalB;

                xvalNext(i) = xval(i) - eps / Scalar{2};
                objective(xvalNext, fvalA);

                xvalNext(i) = xval(i) + eps / Scalar{2};
                objective(xvalNext, fvalB);

                jacobian.col(i) = (fvalB - fvalA) / eps;
            }
        }
    };

    /** Parametrization container for jacobian estimation using finite differences.
      * The method parameter determines the actual finite differences method, which is used
      * for computation.
      * This class holds parameters for the finite differences operation, such as the
      * number of threads to be used and the numerical espilon. */
    template<typename _Scalar, typename _Method>
    struct FiniteDifferences
    {
    public:
        using Scalar = _Scalar;
        using Method = _Method;

        static_assert(Eigen::NumTraits<Scalar>::IsInteger == 0, "Finite differences only supports non-integer scalars.");

        FiniteDifferences() = default;

        FiniteDifferences(const Scalar eps)
            : _eps(eps)
        { }

        FiniteDifferences(const Scalar eps, const int threads)
            : _eps(eps), _threads(threads)
        { }

        /** Sets the numerical espilon that is used as step between two sucessive function evaluations.
          * @param eps numerical epsilon */
        void setEpsilon(const Scalar eps)
        {
            _eps = eps;
        }

        /** Returns the numerical epsilon, which is used for jacobian approximation.
          * @return numerical epsilon */
        Scalar epsilon() const
        {
            return _eps;
        }

        /** Sets the number of threads which should be used to compute finite differences
          * (OMP only).
          * Set to 0 or negative for auto-detection of a suitable number.
          * Each dimension of the input vector can be handled independently.
          * @param threads number of threads */
        void setThreads(const int threads)
        {
            _threads = threads;
        }

        int threads() const
        {
            return _threads;
        }

        /** Executes the chosen finite differences method with the configured parameters.
          * @param xval current state vector of estimation problem.
          * @param fval evaluated residual of the objective at the current state vector.
          * @param objective objective function of the estimation problem
          * @param jacobian jacobian that will be computed by finit differenes */
        template<typename Scalar, int Inputs, int Outputs, typename Objective>
        void operator()(const Eigen::Matrix<Scalar, Inputs, 1> &xval,
                        const Eigen::Matrix<Scalar, Outputs, 1> &fval,
                        const Objective &objective,
                        Eigen::Matrix<Scalar, Outputs, Inputs> &jacobian) const
        {
            _method(xval, fval, objective, _eps, _threads, jacobian);
        }

    private:
        Scalar _eps = std::sqrt(Eigen::NumTraits<Scalar>::epsilon());
        int _threads = int{1};
        Method _method = {};
    };

    /** Generic class for refining a computed newton step.
      * The method parameter determines how the step is actually refined, e.g
      * ArmijoBacktracking, DoglegMethod or WolfeBacktracking. */
    template<typename Scalar, int Inputs, int Outputs, typename Method>
    class NewtonStepRefiner { };

    /** Newton step refinement method which applies a constant scaling factor to the newton step. */
    struct ConstantStepFactor { };

    template<typename _Scalar, int _Inputs, int _Outputs>
    class NewtonStepRefiner<_Scalar, _Inputs, _Outputs, ConstantStepFactor>
    {
    public:
        using Scalar = _Scalar;
        static constexpr int Inputs = _Inputs;
        static constexpr int Outputs = _Outputs;
        using Method = ConstantStepFactor;

        static_assert(Eigen::NumTraits<Scalar>::IsInteger == 0, "Step refinement only supports non-integer scalars");

        using InputVector = Eigen::Matrix<Scalar, Inputs, 1>;
        using OutputVector = Eigen::Matrix<Scalar, Outputs, 1>;
        using JacobiMatrix = Eigen::Matrix<Scalar, Outputs, Inputs>;
        using GradientVector = Eigen::Matrix<Scalar, Inputs, 1>;
        using StepVector = Eigen::Matrix<Scalar, Inputs, 1>;

        NewtonStepRefiner() = default;

        NewtonStepRefiner(const Scalar factor)
            : _factor(factor)
        { }

        /** Sets the constant scaling factor which is applied to the newton step.
          * @param factor constant newton step scaling factor */
        void setFactor(const Scalar factor)
        {
            _factor = factor;
        }

        /** Returns the constant scaling factor which is applied to the newton step.
          * @return constant newton step scaling factor */
        Scalar factor() const
        {
            return _factor;
        }

        /** Refines the given newton step and scales it by a constant factor.
          * @param step newton step which is scaled. */
        template<typename Objective>
        void operator()(const InputVector &,
                        const OutputVector &,
                        const JacobiMatrix &,
                        const GradientVector &,
                        const Objective &,
                        StepVector &step) const
        {
            step *= _factor;
        }
    private:
        Scalar _factor = Scalar{1};
    };

    /** Applies Barzilai-Borwein (BB) refinemnt to the newton step.
      * The functor can either compute the direct or inverse BB step.
      * The steps are computed as follows:
      *
      * s_k = x_k - x_k-1         k >= 1
      * y_k = step_k - step_k-1   k >= 1
      * Direct:  stepSize = (s_k^T * s_k) / (y_k^T * s_k)
      * Inverse: stepSize = (y_k^T * s_k) / (y_k^T * y_k)
      *
      * The very first step is computed as a constant. */
    struct BarzilaiBorwein
    {
        enum class Mode
        {
            Direct,
            Inverse
        };
    };

    template<typename _Scalar, int _Inputs, int _Outputs>
    class NewtonStepRefiner<_Scalar, _Inputs, _Outputs, BarzilaiBorwein>
    {
    public:
        using Scalar = _Scalar;
        static constexpr int Inputs = _Inputs;
        static constexpr int Outputs = _Outputs;
        using Method = BarzilaiBorwein;

        static_assert(Eigen::NumTraits<Scalar>::IsInteger == 0, "Step refinement only supports non-integer scalars");

        using InputVector = Eigen::Matrix<Scalar, Inputs, 1>;
        using OutputVector = Eigen::Matrix<Scalar, Outputs, 1>;
        using JacobiMatrix = Eigen::Matrix<Scalar, Outputs, Inputs>;
        using GradientVector = Eigen::Matrix<Scalar, Inputs, 1>;
        using StepVector = Eigen::Matrix<Scalar, Inputs, 1>;
        using Mode = BarzilaiBorwein::Mode;

        NewtonStepRefiner()
        {
            init();
        }

        NewtonStepRefiner(const Mode mode)
            : _mode(mode)
        {
            init();
        }

        NewtonStepRefiner(const Scalar constStep)
            : _constStep(constStep)
        {
            init();
        }

        NewtonStepRefiner(const Mode mode, const Scalar constStep)
            : _mode(mode), _constStep(constStep)
        {
            init();
        }

        /** Sets the BarzilaiBorwein operation mode.
          * @param mode mode */
        void setMode(const Mode mode)
        {
            _mode = mode;
        }

        /** Returns the BarzilaiBorwein operation mode.
          * @return mode */
        Mode mode() const
        {
            return _mode;
        }

        /** Sets the constant step size, which is used when the refiner was not initialized yet.
          * @param stepSize constant step size */
        void setConstantStepSize(const Scalar stepSize)
        {
            _constStep = stepSize;
        }

        /** Returns he constant step size, which is used when the refiner was not initialized yet.
          * @return constant step size */
        Scalar constantStepSize() const
        {
            return _constStep;
        }

        template<typename Objective>
        void operator()(const InputVector &xval,
            const OutputVector &,
            const JacobiMatrix &,
            const GradientVector &,
            const Objective &,
            StepVector &step)
        {
            auto stepSize = Scalar{0};
            if(_lastXval.sum() == Scalar{0})
            {
                stepSize = (Scalar{1} / step.norm()) * _constStep;
            }
            else
            {
                switch(_mode)
                {
                case Mode::Direct:
                    stepSize = directStep(xval, step);
                    break;
                case Mode::Inverse:
                    stepSize = inverseStep(xval, step);
                    break;
                default:
                    assert(false);
                    break;
                }
            }

            _lastStep = step;
            _lastXval = xval;

            step *= stepSize;
        }
    private:
        InputVector _lastXval = {};
        StepVector _lastStep = {};
        Mode _mode = Mode::Direct;
        Scalar _constStep = static_cast<Scalar>(1e-2);

        void init()
        {
            _lastXval.setZero();
            _lastStep.setZero();
        }

        Scalar directStep(const InputVector &xval,
                          const StepVector &step) const
        {
            const auto sk = xval - _lastXval;
            const auto yk = step - _lastStep;
            const auto num = sk.dot(sk);
            const auto denom = sk.dot(yk);

            if(denom == Scalar{0})
                return Scalar{1};
            else
                return num / denom;
        }

        Scalar inverseStep(const InputVector &xval,
                           const StepVector &step) const
        {
            const auto sk = xval - _lastXval;
            const auto yk = step - _lastStep;
            const auto num = sk.dot(yk);
            const auto denom = yk.dot(yk);

            if(denom == Scalar{0})
                return Scalar{1};
            else
                return num / denom;
        }
    };

    /** Step size functor to perform Armijo Linesearch with backtracking.
      * The functor iteratively decreases the step size until the following
      * conditions are met:
      *
      * Armijo: f(x - stepSize * grad(x)) <= f(x) - c1 * stepSize * grad(x)^T * grad(x)
      *
      * If the condition does not hold the step size is decreased:
      *
      * stepSize = decrease * stepSize */
    struct ArmijoBacktracking { };

    template<typename _Scalar, int _Inputs, int _Outputs>
    class NewtonStepRefiner<_Scalar, _Inputs, _Outputs, ArmijoBacktracking>
    {
    public:
        using Scalar = _Scalar;
        static constexpr int Inputs = _Inputs;
        static constexpr int Outputs = _Outputs;
        using Method = ArmijoBacktracking;

        static_assert(Eigen::NumTraits<Scalar>::IsInteger == 0, "Step refinement only supports non-integer scalars");

        using InputVector = Eigen::Matrix<Scalar, Inputs, 1>;
        using OutputVector = Eigen::Matrix<Scalar, Outputs, 1>;
        using JacobiMatrix = Eigen::Matrix<Scalar, Outputs, Inputs>;
        using GradientVector = Eigen::Matrix<Scalar, Inputs, 1>;
        using StepVector = Eigen::Matrix<Scalar, Inputs, 1>;

        NewtonStepRefiner() = default;

        NewtonStepRefiner(const Scalar decrease,
                          const Scalar c1,
                          const Scalar minStep,
                          const Scalar maxStep,
                          const Index iterations)
            : _decrease(decrease), _c1(c1), _minStep(minStep),
            _maxStep(maxStep), _maxIt(iterations)
        {
            assert(decrease > static_cast<Scalar>(0));
            assert(decrease < static_cast<Scalar>(1));
            assert(c1 > static_cast<Scalar>(0));
            assert(c1 < static_cast<Scalar>(0.5));
        }

        /** Set the decreasing factor for backtracking.
          * Assure that decrease in (0, 1).
          * @param decrease decreasing factor */
        void setBacktrackingDecrease(const Scalar decrease)
        {
            assert(decrease > static_cast<Scalar>(0));
            assert(decrease < static_cast<Scalar>(1));
            _decrease = decrease;
        }

        /** Returns the decreasing factor for backtracking.
          * The value should always lie within (0, 1).
          * @return backtracking decrease */
        Scalar backtrackingDecrease() const
        {
            return _decrease;
        }

        /** Set the relaxation constant for the Armijo condition (see class description).
          * Typically c1 is chosen to be quite small, e.g. 1e-4.
          * Assure that c1 in (0, 0.5).
          * @param c1 armijo constant */
        void setArmijoConstant(const Scalar c1)
        {
            assert(c1 > static_cast<Scalar>(0));
            assert(c1 < static_cast<Scalar>(0.5));
            _c1 = c1;
        }

        /** Returns the the relaxation constant for the Armijo condition (see class description).
          * The value should always lie within (0, 0.5).
          * @return armijo constant */
        Scalar armijoConstant() const
        {
            return _c1;
        }

        /** Set the bounds for the step size during linesearch.
          * The final step size is guaranteed to be in [minStep, maxStep].
          * @param minStep minimum step size
          * @param maxStep maximum step size */
        void setStepBounds(const Scalar minStep, const Scalar maxStep)
        {
            assert(minStep < maxStep);
            _minStep = minStep;
            _maxStep = maxStep;
        }

        /** Returns the minimum bound for the step size during linesearch.
          * @return minimum step size bound */
        Scalar minimumStepBound() const
        {
            return _minStep;
        }

        /** Returns the maximum bound for the step size during linesearch.
          * @return maximum step size bound */
        Scalar maximumStepBound() const
        {
            return _maxStep;
        }

        /** Set the maximum number of iterations.
          * Set to 0 or negative for infinite iterations.
          * @param iterations maximum number of iterations */
        void setMaximumIterations(const Index iterations)
        {
            _maxIt = iterations;
        }

        /** Returns the maximum number of iterations.
          * A value of 0 or negative means infinite iterations.
          * @return maximum number of iterations */
        Index maximumIterations() const
        {
            return _maxIt;
        }

        template<typename Objective>
        void operator()(const InputVector &xval,
                        const OutputVector &fval,
                        const JacobiMatrix &,
                        const GradientVector &gradient,
                        const Objective &objective,
                        StepVector &step) const
        {
            auto stepSize = _maxStep / _decrease;
            JacobiMatrix jacobianN;
            GradientVector gradientN;
            InputVector xvalN;
            OutputVector fvalN;

            const auto error = static_cast<Scalar>(0.5) * fval.squaredNorm();
            const auto stepGrad = gradient.dot(step);
            bool armijoCondition = false;

            auto iterations = Index{0};
            while((_maxIt <= Index{0} || iterations < _maxIt) &&
                   stepSize * _decrease >= _minStep &&
                   !armijoCondition)
            {
                stepSize = _decrease * stepSize;
                xvalN = xval - stepSize * step;
                objective(xvalN, fvalN, jacobianN);
                const auto errorN = static_cast<Scalar>(0.5) * fvalN.squaredNorm();
                gradientN = jacobianN.transpose() * fvalN;

                armijoCondition = errorN <= error + _c1 * stepSize * stepGrad;

                ++iterations;
            }

            step *= stepSize;
        }
    private:
        Scalar _decrease = static_cast<Scalar>(0.8);
        Scalar _c1 = static_cast<Scalar>(1e-4);
        Scalar _minStep = static_cast<Scalar>(1e-10);
        Scalar _maxStep = static_cast<Scalar>(1);
        Index _maxIt = Index{0};
    };

    /** Step size functor to perform Wolfe Linesearch with backtracking.
      * The functor iteratively decreases the step size until the following
      * conditions are met:
      *
      * Armijo: f(x - stepSize * grad(x)) <= f(x) - c1 * stepSize * grad(x)^T * grad(x)
      * Wolfe: grad(x)^T grad(x - stepSize * grad(x)) <= c2 * grad(x)^T * grad(x)
      *
      * If either condition does not hold the step size is decreased:
      *
      * stepSize = decrease * stepSize */
    struct WolfeBacktracking { };

    template<typename Scalar, int Inputs, int Outputs>
    class NewtonStepRefiner<Scalar, Inputs, Outputs, WolfeBacktracking>
    {
    public:
        using InputVector = Eigen::Matrix<Scalar, Inputs, 1>;
        using OutputVector = Eigen::Matrix<Scalar, Outputs, 1>;
        using JacobiMatrix = Eigen::Matrix<Scalar, Outputs, Inputs>;
        using GradientVector = Eigen::Matrix<Scalar, Inputs, 1>;
        using StepVector = Eigen::Matrix<Scalar, Inputs, 1>;

        NewtonStepRefiner() = default;

        NewtonStepRefiner(const Scalar decrease,
            const Scalar c1,
            const Scalar c2,
            const Scalar minStep,
            const Scalar maxStep,
            const Index iterations)
            : _decrease(decrease), _c1(c1), _c2(c2), _minStep(minStep),
            _maxStep(maxStep), _maxIt(iterations)
        { }

        /** Set the decreasing factor for backtracking.
          * Assure that decrease in (0, 1).
          * @param decrease decreasing factor */
        void setBacktrackingDecrease(const Scalar decrease)
        {
            _decrease = decrease;
        }

        /** Set the wolfe constants for Armijo and Wolfe condition (see class
          * description).
          * Assure that c1 < c2 < 1 and c1 in (0, 0.5).
          * Typically c1 is chosen to be quite small, e.g. 1e-4.
          * @param c1 armijo constant
          * @param c2 wolfe constant */
        void setWolfeConstants(const Scalar c1, const Scalar c2)
        {
            assert(c1 > static_cast<Scalar>(0));
            assert(c1 < static_cast<Scalar>(0.5));
            assert(c1 < c2);
            assert(c2 < static_cast<Scalar>(1));
            _c1 = c1;
            _c2 = c2;
        }

        /** Set the bounds for the step size during linesearch.
          * The final step size is guaranteed to be in [minStep, maxStep].
          * @param minStep minimum step size
          * @param maxStep maximum step size */
        void setStepBounds(const Scalar minStep, const Scalar maxStep)
        {
            assert(minStep < maxStep);
            _minStep = minStep;
            _maxStep = maxStep;
        }

        /** Set the maximum number of iterations.
          * Set to 0 or negative for infinite iterations.
          * @param iterations maximum number of iterations */
        void setMaxIterations(const Index iterations)
        {
            _maxIt = iterations;
        }

        template<typename Objective>
        void operator()(const InputVector &xval,
                        const OutputVector &fval,
                        const JacobiMatrix &,
                        const GradientVector &gradient,
                        const Objective &objective,
                        StepVector &step) const
        {

            auto stepSize = _maxStep / _decrease;
            JacobiMatrix jacobianN;
            GradientVector gradientN;
            InputVector xvalN;
            OutputVector fvalN;

            const Scalar error = fval.squaredNorm() / Scalar{2};
            const Scalar stepGrad = gradient.dot(step);
            bool armijoCondition = false;
            bool wolfeCondition = false;

            Index iterations = 0;
            while((_maxIt <= 0 || iterations < _maxIt) &&
                   stepSize * _decrease >= _minStep &&
                   !(armijoCondition && wolfeCondition))
            {
                stepSize = _decrease * stepSize;
                xvalN = xval - stepSize * step;
                objective_(xvalN, fvalN, jacobianN);
                Scalar errorN = fvalN.squaredNorm() / 2;
                gradientN = jacobianN.transpose() * fvalN;

                armijoCondition = errorN <= error + _c1 * stepSize * stepGrad;
                wolfeCondition = gradientN.dot(step) >= _c2 * stepGrad;

                ++iterations;
            }

            step *= stepSize;
        }
    private:
        Scalar _decrease = static_cast<Scalar>(0.8);
        Scalar _c1 = static_cast<Scalar>(1e-4);
        Scalar _c2 = static_cast<Scalar>(0.9);
        Scalar _minStep = static_cast<Scalar>(1e-10);
        Scalar _maxStep = static_cast<Scalar>(1.0);
        Index _maxIt = 0;
    };

    struct DoglegMethod { };

    /** Implementation of Powell's Dogleg Method. */
    template<typename Scalar, int Inputs, int Outputs>
    class NewtonStepRefiner<Scalar, Inputs, Outputs, DoglegMethod>
    {
    public:
        using InputVector = Eigen::Matrix<Scalar, Inputs, 1>;
        using OutputVector = Eigen::Matrix<Scalar, Outputs, 1>;
        using JacobiMatrix = Eigen::Matrix<Scalar, Outputs, Inputs>;
        using HessianMatrix = Eigen::Matrix<Scalar, Inputs, Inputs>;
        using GradientVector = Eigen::Matrix<Scalar, Inputs, 1>;
        using StepVector = Eigen::Matrix<Scalar, Inputs, 1>;

        NewtonStepRefiner() = default;

        NewtonStepRefiner(const Scalar radius,
                          const Scalar maxRadius,
                          const Scalar radiusEps,
                          const Scalar acceptFitness,
                          const Index iterations)
            : _radius (radius), _maxRadius(maxRadius), _radiusEps(radiusEps),
            _acceptFitness(acceptFitness), _maxIt(iterations)
        { }

        /** Set maximum iterations of the trust region radius search.
          * Set to 0 or negative for infinite iterations.
          * @param iterations maximum iterations for radius search */
        void setMaxIterations(const Index iterations)
        {
            _maxIt = iterations;
        }

        /** Set the minimum fitness value at which a model is accepted.
          * The model fitness is computed as follows:
          *
          * fitness = f(xval) - f(xval + step) / m(0) - m(step)
          *
          * Where f(x) is the objective error function and m(x) is the
          * model function describe by the trust region method.
          *
          * @param fitness minimum fitness for step acceptance */
        void setAcceptanceFitness(const Scalar fitness)
        {
            _acceptFitness = fitness;
        }

        /** Set the comparison epsilon on how close the step should be
          * to the trust region radius to trigger an increase of the radius.
          * Should usually be picked low, e.g. 1e-8.
          * @param eps comparison epsilon for radius increase */
        void setRaidusEps(const Scalar eps)
        {
            _radiusEps = eps;
        }

        template<typename Objective>
        void operator()(const InputVector &xval,
                        const OutputVector &fval,
                        const JacobiMatrix &jacobian,
                        const GradientVector &gradient,
                        const Objective &objective,
                        StepVector &step)
        {
            // approximate hessian
            const HessianMatrix hessian = jacobian.transpose() * jacobian;

            // precompute the full step length
            const StepVector fullStep = step;
            const auto fullStepLenSq = fullStep.squaredNorm();

            // compute the cauchy step
            const auto gradientLenSq = gradient.squaredNorm();
            const Scalar curvature = gradient.dot(hessian * gradient);
            const StepVector cauchyStep = -(gradientLenSq / curvature) * gradient;
            const auto cauchyStepLenSq = cauchyStep.squaredNorm();

            // compute step diff
            const StepVector diffStep = fullStep - cauchyStep;
            const auto diffLenSq = diffStep.squaredNorm();
            const Scalar diffFac = cauchyStep.dot(diffStep) / diffLenSq;

            auto modelFitness = _acceptFitness - Scalar{1};
            Index iteration = 0;

            // keep computing while the model fitness is bad
            while(modelFitness < _acceptFitness &&
                  (_maxIt <= 0 || iteration < _maxIt))
            {
                const auto radiusSq = _radius * _radius;

                // if the full step is within the trust region simply
                // use it, it provides a good minimizer
                if(fullStepLenSq <= radiusSq)
                {
                    step = fullStep;
                }
                else
                {
                    // if the cauchy step lies outside the trust region
                    // go towards it until the trust region boundary
                    if(cauchyStepLenSq >= radiusSq)
                    {
                        step = (_radius / std::sqrt(cauchyStepLenSq)) * cauchyStep;
                    }
                    else
                    {
                        const auto secondTerm = std::sqrt(diffFac * diffFac + (radiusSq + cauchyStepLenSq) / diffLenSq);
                        const auto scale1 = -diffFac - secondTerm;
                        const auto scale2 = -diffFac + secondTerm;

                        step = cauchyStep + std::max(scale1, scale2) * (fullStep - cauchyStep);
                    }
                }

                // compute the model fitness to determine the update scheme for
                // the trust region radius
                modelFitness = calulateModelFitness(xval, fval, gradient, hessian, step, objective);

                const auto stepLen = step.norm();

                // if the model fitness is really bad reduce the radius!
                if(modelFitness < static_cast<Scalar>(0.25))
                {
                    _radius = static_cast<Scalar>(0.25) * stepLen;
                }
                // if the model fitness is very good then increase it
                else if(modelFitness > static_cast<Scalar>(0.75) && std::abs(stepLen - _radius) < _radiusEps)
                {
                    // use the double radius
                    _radius = 2 * _radius;
                    // maintain radius border if configured
                    if(_maxRadius > 0 && _radius > _maxRadius)
                        _radius = _maxRadius;
                }

                ++iteration;
            }
        }

    private:
        Scalar _radius = Scalar{1};
        Scalar _maxRadius = Scalar{2};
        Scalar _radiusEps = static_cast<Scalar>(1e-6);
        Scalar _acceptFitness = static_cast<Scalar>(0.25);
        Index _maxIt = 0;

        template<typename Objective>
        Scalar calulateModelFitness(const InputVector &xval,
                                    const OutputVector &fval,
                                    const GradientVector &gradient,
                                    const HessianMatrix &hessian,
                                    const StepVector &step,
                                    const Objective &objective) const
        {
            const Scalar error = fval.squaredNorm() / Scalar{2};

            // evaluate the error function at the new position
            InputVector xvalNext = xval + step;
            OutputVector fvalNext;
            JacobiMatrix jacobianNext;
            objective(xvalNext, fvalNext, jacobianNext);
            // compute the actual new error
            const Scalar nextError = fvalNext.squaredNorm() / Scalar{2};
            // compute the new error by the model
            const Scalar modelError = error + gradient.dot(step) + step.dot(hessian * step) / Scalar{2};

            return (error - nextError) / (error - modelError);
        }
    };

    struct DenseSVDSolver
    {
        template<typename DerivedA, typename DerivedB>
        auto operator()(const Eigen::MatrixBase<DerivedA> &A, const Eigen::MatrixBase<DerivedB> &b) const
        {
            using Matrix = typename Eigen::MatrixBase<DerivedA>::PlainMatrix;
            using Solver = Eigen::JacobiSVD<Matrix, Eigen::FullPivHouseholderQRPreconditioner>;
            auto solver = Solver(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
            return solver.solve(b);
        }
    };

    struct DenseCholeskySolver
    {
        template<typename DerivedA, typename DerivedB>
        auto operator()(const Eigen::MatrixBase<DerivedA> &A, const Eigen::MatrixBase<DerivedB> &b) const
        {
            using Matrix = typename Eigen::MatrixBase<DerivedA>::PlainMatrix;
            using Solver = Eigen::LDLT<Matrix>;

            Solver decomp;
            decomp.compute(A);

            if(!decomp.isPositive())
                throw std::runtime_error("DenseCholeskySolver: matrix is not positive semi-definite");

            return decomp.solve(b);
        }
    };

    struct GradientDescentMethod
    {
        template<typename Scalar, int Inputs, int Outputs>
        auto operator()(const Eigen::Matrix<Scalar, Inputs, 1>&,
                        const Eigen::Matrix<Scalar, Outputs, 1> &,
                        const Eigen::Matrix<Scalar, Outputs, Inputs> &,
                        const Eigen::Matrix<Scalar, Inputs, 1> &gradient) const
        {
            return gradient;
        }
    };

    template<typename Solver=DenseSVDSolver>
    struct GaussNewtonMethod
    {
        template<typename Scalar, int Inputs, int Outputs>
        auto operator()(const Eigen::Matrix<Scalar, Inputs, 1>&,
                        const Eigen::Matrix<Scalar, Outputs, 1> &,
                        const Eigen::Matrix<Scalar, Outputs, Inputs> &jacobian,
                        const Eigen::Matrix<Scalar, Inputs, 1> &gradient) const
        {
            using Matrix = Eigen::Matrix<Scalar, Inputs, Inputs>;

            Solver solver;

            Matrix A = jacobian.transpose() * jacobian;
            return solver(A, gradient);
        }
    };

    namespace internal
    {
        template<bool ComputesJacobian>
        struct ObjectiveEvaluator { };

        template<>
        struct ObjectiveEvaluator<true>
        {
            template<typename InputVector, typename Objective, typename FiniteDifferencesCalculator, typename OutputVector, typename JacobiMatrix>
            void operator()(const InputVector &xval,
                          const Objective& objective,
                          const FiniteDifferencesCalculator&,
                          OutputVector &fval,
                          JacobiMatrix &jacobian) const
            {
                objective(xval, fval, jacobian);
            }
        };

        template<>
        struct ObjectiveEvaluator<false>
        {
            template<typename InputVector, typename Objective, typename FiniteDifferencesCalculator, typename OutputVector, typename JacobiMatrix>
            void operator()(const InputVector &xval,
                          const Objective& objective,
                          const FiniteDifferencesCalculator& finiteDifferences,
                          OutputVector &fval,
                          JacobiMatrix &jacobian) const
            {
                objective(xval, fval);
                finiteDifferences(xval, fval, jacobian);
            }
        };
    }

    /** Base class for least squares algorithms.
      * It implements the whole optimization strategy except the step
      * calculation. Cannot be instantiated. */
    template<typename _Scalar,
             int _Inputs,
             int _Outputs,
             typename _Objective,
             typename _StepMethod,
             typename _RefineMethod,
             typename _FiniteDifferencesMethod>
    class LeastSquaresAlgorithm
    {
    public:
        using Scalar = _Scalar;
        constexpr static auto Inputs = _Inputs;
        constexpr static auto Outputs = _Outputs;
        using Objective = _Objective;
        using StepMethod = _StepMethod;

        using InputVector = Eigen::Matrix<Scalar, Inputs, 1>;
        using OutputVector = Eigen::Matrix<Scalar, Outputs, 1>;
        using JacobiMatrix = Eigen::Matrix<Scalar, Outputs, Inputs>;
        using HessianMatrix = Eigen::Matrix<Scalar, Inputs, Inputs>;
        using GradientVector = Eigen::Matrix<Scalar, Inputs, 1>;
        using StepVector = Eigen::Matrix<Scalar, Inputs, 1>;

        constexpr static auto ComputesJacobian = std::is_invocable<void(), const InputVector&, OutputVector&, JacobiMatrix&>::value;

        using FiniteDifferencesCalculator = FiniteDifferences<Scalar, _FiniteDifferencesMethod>;
        using StepRefiner = NewtonStepRefiner<Scalar, Inputs, Outputs, _RefineMethod>;
        using Callback = std::function<bool(const Index,
                                            const InputVector&,
                                            const OutputVector&,
                                            const JacobiMatrix&,
                                            const GradientVector&,
                                            const StepVector&)>;

        struct Result
        {
            InputVector xval;
            OutputVector fval;
            Scalar error;
            Index iterations;
            bool converged;
        };

        LeastSquaresAlgorithm() = default;

        /** Set the number of threads used to compute gradients.
          * This only works if OpenMP is enabled.
          * Set to 0 to allow automatic detection of thread number.
          * @param threads number of threads to be used */
        void setThreads(const Index threads)
        {
            _finiteDifferences.setThreads(threads);
        }

        /** Set the difference for gradient estimation with finite differences.
          * @param eps numerical epsilon */
        void setNumericalEpsilon(const Scalar eps)
        {
            _finiteDifferences.setNumericalEpsilon(eps);
        }

        /** Sets the instance values of the custom objective function.
          * Should be used if the objective function requires custom data parameters.
          * @param objective instance that should be copied */
        void setObjective(const Objective &objective)
        {
            _objective = objective;
        }

        void setCallback(const Callback &callback)
        {
            _callback = callback;
        }

        /** Sets the instance values of the step refiner functor.
          * @param refiner instance that should be copied */
        void setStepRefiner(const StepRefiner &refiner)
        {
            _stepRefiner = refiner;
        }

        /** Set the maximum number of iterations.
          * Set to 0 or negative for infinite iterations.
          * @param iterations maximum number of iterations */
        void setMaxIterations(const Index iterations)
        {
            _maxIt = iterations;
        }

        /** Set the minimum step length between two iterations.
          * If the step length falls below this value, the optimizer stops.
          * @param steplen minimum step length */
        void setMinStepLength(const Scalar steplen)
        {
            _minStepLen = steplen;
        }

        /** Set the minimum gradient length.
          * If the gradient length falls below this value, the optimizer stops.
          * @param gradlen minimum gradient length */
        void setMinGradientLength(const Scalar gradlen)
        {
            _minGradLen = gradlen;
        }

        /** Set the minimum squared error.
          * If the error falls below this value, the optimizer stops.
          * @param error minimum error */
        void setMinError(const Scalar error)
        {
            _minError = error;
        }

        /** Set the level of verbosity to print status information after each
          * iteration.
          * Set to 0 to deacticate any output.
          * @param verbosity level of verbosity */
        void setVerbosity(const Index verbosity)
        {
            _verbosity = verbosity;
        }

        Result minimize(const InputVector &initialGuess)
        {
            InputVector xval = initialGuess;
            OutputVector fval;
            JacobiMatrix jacobian;
            GradientVector gradient;
            StepVector step = StepVector::Zero(xval.size());

            auto gradLen = _minGradLen + 1;
            auto error = _minError + 1;
            auto stepLen = _minStepLen + 1;
            bool callbackResult = true;

            const auto objective = internal::ObjectiveEvaluator<ComputesJacobian>();

            Index iterations = 0;
            while((_maxIt <= 0 || iterations < _maxIt) &&
                   gradLen >= _minGradLen &&
                   stepLen >= _minStepLen &&
                   error >= _minError &&
                   callbackResult)
            {
                xval -= step;
                objective(xval, fval, jacobian);

                error = fval.squaredNorm() / 2;
                gradient = jacobian.transpose() * fval;
                gradLen = gradient.norm();

                // compute the full newton step according to the current method
                step = _stepMethod(xval, fval, jacobian, gradient);

                // refine the step according to the current refiner
                _stepRefiner(xval, fval, jacobian, gradient, objective, step);
                stepLen = step.norm();

                // evaluate callback if available
                if(_callback)
                {
                    callbackResult = _callback(iterations + 1, xval, fval, jacobian, gradient, step);
                }

                if(_verbosity > 0)
                {
                    std::stringstream ss;
                    ss << "it=" << std::setfill('0')
                        << std::setw(4) << iterations
                        << std::fixed << std::showpoint << std::setprecision(6)
                        << "    steplen=" << stepLen
                        << "    gradlen=" << gradLen;

                    if(_verbosity > 1)
                        ss << "    callback=" << (callbackResult ? "true" : "false");

                    ss << "    error=" << error;

                    if(_verbosity > 2)
                        ss << "    fval=" << vector2str(fval);
                    if(_verbosity > 3)
                        ss << "    xval=" << vector2str(xval);
                    if(_verbosity > 4)
                        ss << "    step=" << vector2str(step);
                    std::cout << ss.str() << std::endl;
                }

                ++iterations;
            }

            Result result;
            result.xval = xval;
            result.fval = fval;
            result.error = error;
            result.iterations = iterations;
            result.converged = stepLen < _minStepLen ||
                gradLen < _minGradLen ||
                error < _minError;

            return result;
        }

    private:
        Objective _objective = {};
        StepMethod _stepMethod = {};
        Callback _callback = {};
        FiniteDifferencesCalculator _finiteDifferences = {};
        StepRefiner _stepRefiner = {};

        Index _maxIt = 0;
        Scalar _minStepLen = static_cast<Scalar>(1e-9);
        Scalar _minGradLen = static_cast<Scalar>(1e-9);
        Scalar _minError = Scalar{0};
        Index _verbosity = 0;

        template<typename Derived>
        std::string vector2str(const Eigen::MatrixBase<Derived> &vec) const
        {
            assert(vec.cols() == 1);

            std::stringstream ss1;
            ss1 << std::fixed << std::showpoint << std::setprecision(6);
            std::stringstream ss2;
            ss2 << '[';
            for(Index i = 0; i < vec.rows(); ++i)
            {
                ss1 << vec(i, 0);
                ss2 << std::setfill(' ') << std::setw(10) << ss1.str();
                if(i != vec.rows() - 1)
                    ss2 << ' ';
                ss1.str("");
            }
            ss2 << ']';

            return ss2.str();
        }
    };

    // template<typename Scalar,
    //     typename ErrorFunction,
    //     typename Callback=NoCallback<Scalar>,
    //     typename FiniteDifferences=CentralDifferences<Scalar>,
    //     typename Solver=DenseSVDSolver<Scalar>>
    // class LevenbergMarquardt : public LeastSquaresAlgorithm<Scalar, ErrorFunction,
    //     ConstantStepSize<Scalar>, Callback, FiniteDifferences>
    // {
    // public:
    //     typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    //     typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    // private:
    //     Scalar increase_;
    //     Scalar decrease_;
    //     Scalar lambda_;
    //     Index maxItLM_;

    // public:
    //     LevenbergMarquardt()
    //         : LeastSquaresAlgorithm<Scalar, ErrorFunction,
    //             ConstantStepSize<Scalar>, Callback, FiniteDifferences>(),
    //             increase_(static_cast<Scalar>(2)),
    //             decrease_(static_cast<Scalar>(0.5)),
    //             lambda_(static_cast<Scalar>(1)),
    //             maxItLM_(0)
    //     { }

    //     /** Set the initial gradient descent factor of levenberg marquardt.
    //       * @param lambda gradient descent factor */
    //     void setLambda(const Scalar lambda)
    //     {
    //         lambda_ = lambda;
    //     }

    //     /** Set maximum iterations of the levenberg marquardt optimization.
    //       * Set to 0 or negative for infinite iterations.
    //       * @param iterations maximum iterations for lambda search */
    //     void setMaxIterationsLM(const Index iterations)
    //     {
    //         maxItLM_ = iterations;
    //     }

    //     /** Set the increase factor for the lambda damping.
    //       * Make sure the value is greater than 1.
    //       * @param increase factor for increasing lambda */
    //     void setLambdaIncrease(const Scalar increase)
    //     {
    //         assert(increase > static_cast<Scalar>(1));
    //         increase_ = increase;
    //     }

    //     /** Set the decrease factor for the lambda damping.
    //       * Make sure the value is in (0, 1).
    //       * @param increase factor for increasing lambda */
    //     void setLambdaDecrease(const Scalar decrease)
    //     {
    //         assert(decrease < static_cast<Scalar>(1));
    //         assert(decrease > static_cast<Scalar>(0));
    //         decrease_ = decrease;
    //     }

    //     void calculateStep(const Vector &xval,
    //         const Vector &fval,
    //         const Matrix &jacobian,
    //         const Vector &gradient,
    //         Vector &step) override
    //     {
    //         Solver solver;
    //         Scalar error = fval.squaredNorm() / 2;
    //         Scalar errorN = error + 1;

    //         Vector xvalN;
    //         Vector fvalN;
    //         Matrix jacobianN;

    //         Matrix jacobianSq = jacobian.transpose() * jacobian;
    //         Matrix A;

    //         Index iterations = 0;
    //         while((maxItLM_ <= 0 || iterations < maxItLM_) &&
    //             errorN > error)
    //         {
    //             A = jacobianSq;
    //             // add identity matrix
    //             for(Index i = 0; i < A.rows(); ++i)
    //                 A(i, i) += lambda_;

    //             solver(A, gradient, step);

    //             xvalN = xval - step;
    //             this->errorFunction_(xvalN, fvalN, jacobianN);
    //             errorN = fvalN.squaredNorm() / 2;

    //             if(errorN > error)
    //                 lambda_ *= increase_;
    //             else
    //                 lambda_ *= decrease_;

    //             ++iterations;
    //         }
    //     }
    // };
}

#endif
