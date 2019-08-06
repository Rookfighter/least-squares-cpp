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

namespace lsq
{
    typedef long int Index;

    /** Functor to compute forward differences.
      * Computes the gradient of the objective f(x) as follows:
      *
      * grad(x) = (f(x + eps) - f(x)) / eps
      *
      * The computation requires len(x) evaluations of the objective.
      */
    template<typename Scalar>
    class ForwardDifferences
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef std::function<void(const Vector &, Vector &)> ErrorFunction;
    private:
        Scalar eps_;
        Index threads_;
        ErrorFunction objective_;
    public:
        ForwardDifferences()
            : ForwardDifferences(
                std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        { }

        ForwardDifferences(const Scalar eps)
            : eps_(eps), threads_(1), objective_()
        { }

        void setNumericalEpsilon(const Scalar eps)
        {
            eps_ = eps;
        }

        void setThreads(const Index threads)
        {
            threads_ = threads;
        }

        void setErrorFunction(const ErrorFunction &objective)
        {
            objective_ = objective;
        }

        void operator()(const Vector &xval,
            const Vector &fval,
            Matrix &jacobian)
        {
            assert(objective_);

            jacobian.resize(fval.size(), xval.size());
            #pragma omp parallel for num_threads(threads_)
            for(Index i = 0; i < xval.size(); ++i)
            {
                Vector fvalN;
                Vector xvalN = xval;
                xvalN(i) += eps_;
                objective_(xvalN, fvalN);

                jacobian.col(i) = (fvalN - fval) / eps_;
            }
        }
    };

    /** Functor to compute backward differences.
      * Computes the gradient of the objective f(x) as follows:
      *
      * grad(x) = (f(x) - f(x - eps)) / eps
      *
      * The computation requires len(x) evaluations of the objective.
      */
    template<typename Scalar>
    class BackwardDifferences
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef std::function<void(const Vector &, Vector &)> ErrorFunction;
    private:
        Scalar eps_;
        Index threads_;
        ErrorFunction objective_;
    public:
        BackwardDifferences()
            : BackwardDifferences(
                std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        { }

        BackwardDifferences(const Scalar eps)
            : eps_(eps), threads_(1), objective_()
        { }

        void setNumericalEpsilon(const Scalar eps)
        {
            eps_ = eps;
        }

        void setThreads(const Index threads)
        {
            threads_ = threads;
        }

        void setErrorFunction(const ErrorFunction &objective)
        {
            objective_ = objective;
        }

        void operator()(const Vector &xval,
            const Vector &fval,
            Matrix &jacobian)
        {
            assert(objective_);

            jacobian.resize(fval.size(), xval.size());
            #pragma omp parallel for num_threads(threads_)
            for(Index i = 0; i < xval.size(); ++i)
            {
                Vector fvalN;
                Vector xvalN = xval;
                xvalN(i) -= eps_;
                objective_(xvalN, fvalN);
                jacobian.col(i) = (fval - fvalN) / eps_;
            }
        }
    };

    /** Functor to compute central differences.
      * Computes the gradient of the objective f(x) as follows:
      *
      * grad(x) = (f(x + 0.5 eps) - f(x - 0.5 eps)) / eps
      *
      * The computation requires 2 * len(x) evaluations of the objective.
      */
    template<typename Scalar>
    struct CentralDifferences
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef std::function<void(const Vector &, Vector &)> ErrorFunction;
    private:
        Scalar eps_;
        Index threads_;
        ErrorFunction objective_;
    public:
        CentralDifferences()
            : CentralDifferences(
                std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        { }

        CentralDifferences(const Scalar eps)
            : eps_(eps), threads_(1), objective_()
        { }

        void setNumericalEpsilon(const Scalar eps)
        {
            eps_ = eps;
        }

        void setThreads(const Index threads)
        {
            threads_ = threads;
        }

        void setErrorFunction(const ErrorFunction &objective)
        {
            objective_ = objective;
        }

        void operator()(const Vector &xval,
            const Vector &fval,
            Matrix &jacobian)
        {
            assert(objective_);

            std::vector<Vector> fvalN(xval.size() * 2);
            #pragma omp parallel for num_threads(threads_)
            for(Index i = 0; i < static_cast<Index>(fvalN.size()); ++i)
            {
                Index idx = i / 2;
                Vector xvalN = xval;
                if(i % 2 == 0)
                    xvalN(idx) += eps_ / 2;
                else
                    xvalN(idx) -= eps_ / 2;

                objective_(xvalN, fvalN[i]);
            }

            jacobian.resize(fval.size(), xval.size());
            for(Index i = 0; i < xval.size(); ++i)
                jacobian.col(i) = (fvalN[i * 2] - fvalN[i * 2 + 1]) / eps_;
        }
    };

    /** Dummy callback functor, which does nothing. */
    template<typename Scalar>
    struct NoCallback
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

        bool operator()(const Index,
            const Vector &,
            const Vector &,
            const Matrix &,
            const Vector &,
            const Vector &) const
        {
            return true;
        }
    };

    /** Step size functor, which returns a constant step size. */
    template<typename Scalar>
    class ConstantStepSize
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef std::function<void(const Vector &, Vector &, Matrix &)> ErrorFunction;
    private:
        Scalar stepSize_;
    public:

        ConstantStepSize()
            : ConstantStepSize(1.0)
        {

        }

        ConstantStepSize(const Scalar stepSize)
            : stepSize_(stepSize)
        {

        }

        /** Set the step size returned by this functor.
          * @param stepSize step size returned by functor */
        void setStepSize(const Scalar stepSize)
        {
            stepSize_ = stepSize;
        }

        void setErrorFunction(const ErrorFunction &)
        { }

        Scalar operator()(const Vector &,
            const Vector &,
            const Matrix &,
            const Vector &,
            const Vector &)
        {
            return stepSize_;
        }
    };

    /** Step size functor to compute Barzilai-Borwein (BB) steps.
      * The functor can either compute the direct or inverse BB step.
      * The steps are computed as follows:
      *
      * s_k = x_k - x_k-1         k >= 1
      * y_k = step_k - step_k-1   k >= 1
      * Direct:  stepSize = (s_k^T * s_k) / (y_k^T * s_k)
      * Inverse: stepSize = (y_k^T * s_k) / (y_k^T * y_k)
      *
      * The very first step is computed as a constant. */
    template<typename Scalar>
    class BarzilaiBorwein
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef std::function<void(const Vector &, Vector &, Matrix &)> ErrorFunction;

        enum class Method
        {
            Direct,
            Inverse
        };
    private:
        Vector lastXval_;
        Vector lastStep_;
        Method method_;
        Scalar constStep_;

        Scalar constantStep() const
        {
            return constStep_;
        }

        Scalar directStep(const Vector &xval,
            const Vector &step)
        {
            auto sk = xval - lastXval_;
            auto yk = step - lastStep_;
            Scalar num = sk.dot(sk);
            Scalar denom = sk.dot(yk);

            if(denom == 0)
                return 1;
            else
                return num / denom;
        }

        Scalar inverseStep(const Vector &xval,
            const Vector &step)
        {
            auto sk = xval - lastXval_;
            auto yk = step - lastStep_;
            Scalar num = sk.dot(yk);
            Scalar denom = yk.dot(yk);

            if(denom == 0)
                return 1;
            else
                return num / denom;
        }
    public:
        BarzilaiBorwein()
            : BarzilaiBorwein(Method::Direct, 1e-2)
        { }

        BarzilaiBorwein(const Method method, const Scalar constStep)
            : lastXval_(), lastStep_(), method_(method),
            constStep_(constStep)
        { }

        void setErrorFunction(const ErrorFunction &)
        { }

        void setMethod(const Method method)
        {
            method_ = method;
        }

        void setConstStepSize(const Scalar stepSize)
        {
            constStep_ = stepSize;
        }

        Scalar operator()(const Vector &xval,
            const Vector &,
            const Matrix &,
            const Vector &,
            const Vector &step)
        {
            Scalar stepSize = 0;
            if(lastXval_.size() == 0)
            {
                stepSize = (1 / step.norm()) * constStep_;
            }
            else
            {
                switch(method_)
                {
                case Method::Direct:
                    stepSize = directStep(xval, step);
                    break;
                case Method::Inverse:
                    stepSize = inverseStep(xval, step);
                    break;
                default:
                    assert(false);
                    break;
                }
            }

            lastStep_ = step;
            lastXval_ = xval;

            return stepSize;
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
      * stepSize = decrease * stepSize
      */
    template<typename Scalar>
    class ArmijoBacktracking
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef std::function<void(const Vector &, Vector &, Matrix &)> ErrorFunction;
    private:
        Scalar decrease_;
        Scalar c1_;
        Scalar minStep_;
        Scalar maxStep_;
        Index maxIt_;
        ErrorFunction objective_;

    public:
        ArmijoBacktracking()
            : ArmijoBacktracking(0.8, 0.1, 1e-10, 1.0, 0)
        { }

        ArmijoBacktracking(const Scalar decrease,
            const Scalar c1,
            const Scalar minStep,
            const Scalar maxStep,
            const Index iterations)
            : decrease_(decrease), c1_(c1), minStep_(minStep),
            maxStep_(maxStep), maxIt_(iterations), objective_()
        { }

        /** Set the decreasing factor for backtracking.
          * Assure that decrease in (0, 1).
          * @param decrease decreasing factor */
        void setBacktrackingDecrease(const Scalar decrease)
        {
            decrease_ = decrease;
        }

        /** Set the Armijo constant for the Armijo (see class description).
          * Assure that c1 in (0, 0.5).
          * @param c1 armijo constant */
        void setArmijoConstant(const Scalar c1)
        {
            assert(c1 > 0);
            assert(c1 < 0.5);
            c1_ = c1;
        }

        /** Set the bounds for the step size during linesearch.
          * The final step size is guaranteed to be in [minStep, maxStep].
          * @param minStep minimum step size
          * @param maxStep maximum step size */
        void setStepBounds(const Scalar minStep, const Scalar maxStep)
        {
            assert(minStep < maxStep);
            minStep_ = minStep;
            maxStep_ = maxStep;
        }

        /** Set the maximum number of iterations.
          * Set to 0 or negative for infinite iterations.
          * @param iterations maximum number of iterations */
        void setMaxIterations(const Index iterations)
        {
            maxIt_ = iterations;
        }

        void setErrorFunction(const ErrorFunction &objective)
        {
            objective_ = objective;
        }

        Scalar operator()(const Vector &xval,
            const Vector &fval,
            const Matrix &,
            const Vector &gradient,
            const Vector &step)
        {
            assert(objective_);

            Scalar stepSize = maxStep_ / decrease_;
            Matrix jacobianN;
            Vector gradientN;
            Vector xvalN;
            Vector fvalN;

            Scalar error = 0.5 * fval.squaredNorm();
            Scalar stepGrad = gradient.dot(step);
            bool armijoCondition = false;

            Index iterations = 0;
            while((maxIt_ <= 0 || iterations < maxIt_) &&
                stepSize * decrease_ >= minStep_ &&
                !armijoCondition)
            {
                stepSize = decrease_ * stepSize;
                xvalN = xval - stepSize * step;
                objective_(xvalN, fvalN, jacobianN);
                Scalar errorN = 0.5 * fvalN.squaredNorm();
                gradientN = jacobianN.transpose() * fvalN;

                armijoCondition = errorN <= error + c1_ * stepSize * stepGrad;

                ++iterations;
            }

            return stepSize;
        }
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
      * stepSize = decrease * stepSize
      *
      */
    template<typename Scalar>
    class WolfeBacktracking
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef std::function<void(const Vector &, Vector &, Matrix &)> ErrorFunction;
    private:
        Scalar decrease_;
        Scalar c1_;
        Scalar c2_;
        Scalar minStep_;
        Scalar maxStep_;
        Index maxIt_;
        ErrorFunction objective_;

    public:
        WolfeBacktracking()
            : WolfeBacktracking(0.8, 0.1, 0.9, 1e-10, 1.0, 0)
        { }

        WolfeBacktracking(const Scalar decrease,
            const Scalar c1,
            const Scalar c2,
            const Scalar minStep,
            const Scalar maxStep,
            const Index iterations)
            : decrease_(decrease), c1_(c1), c2_(c2), minStep_(minStep),
            maxStep_(maxStep), maxIt_(iterations), objective_()
        { }

        /** Set the decreasing factor for backtracking.
          * Assure that decrease in (0, 1).
          * @param decrease decreasing factor */
        void setBacktrackingDecrease(const Scalar decrease)
        {
            decrease_ = decrease;
        }

        /** Set the wolfe constants for Armijo and Wolfe condition (see class
          * description).
          * Assure that c1 < c2 < 1 and c1 in (0, 0.5).
          * @param c1 armijo constant
          * @param c2 wolfe constant */
        void setWolfeConstants(const Scalar c1, const Scalar c2)
        {
            assert(c1 < c2);
            assert(c2 < 1);
            c1_ = c1;
            c2_ = c2;
        }

        /** Set the bounds for the step size during linesearch.
          * The final step size is guaranteed to be in [minStep, maxStep].
          * @param minStep minimum step size
          * @param maxStep maximum step size */
        void setStepBounds(const Scalar minStep, const Scalar maxStep)
        {
            assert(minStep < maxStep);
            minStep_ = minStep;
            maxStep_ = maxStep;
        }

        /** Set the maximum number of iterations.
          * Set to 0 or negative for infinite iterations.
          * @param iterations maximum number of iterations */
        void setMaxIterations(const Index iterations)
        {
            maxIt_ = iterations;
        }

        void setErrorFunction(const ErrorFunction &objective)
        {
            objective_ = objective;
        }

        Scalar operator()(const Vector &xval,
            const Vector &fval,
            const Matrix &,
            const Vector &gradient,
            const Vector &step)
        {
            assert(objective_);

            Scalar stepSize = maxStep_ / decrease_;
            Matrix jacobianN;
            Vector gradientN;
            Vector xvalN;
            Vector fvalN;

            Scalar error = 0.5 * fval.squaredNorm();
            Scalar stepGrad = gradient.dot(step);
            bool armijoCondition = false;
            bool wolfeCondition = false;

            Index iterations = 0;
            while((maxIt_ <= 0 || iterations < maxIt_) &&
                stepSize * decrease_ >= minStep_ &&
                !(armijoCondition && wolfeCondition))
            {
                stepSize = decrease_ * stepSize;
                xvalN = xval - stepSize * step;
                objective_(xvalN, fvalN, jacobianN);
                Scalar errorN = 0.5 * fvalN.squaredNorm();
                gradientN = jacobianN.transpose() * fvalN;

                armijoCondition = errorN <= error + c1_ * stepSize * stepGrad;
                wolfeCondition = gradientN.dot(step) >= c2_ * stepGrad;

                ++iterations;
            }

            return stepSize;
        }
    };

    template<typename Scalar>
    struct DenseSVDSolver
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::JacobiSVD<Matrix, Eigen::FullPivHouseholderQRPreconditioner>
            Solver;

        void operator()(const Matrix &A, const Vector &b, Vector &result) const
        {
            Solver solver(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
            result = solver.solve(b);
        }
    };

    template<typename Scalar>
    struct DenseCholeskySolver
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::LDLT<Matrix> Solver;

        void operator()(const Matrix &A, const Vector &b, Vector &result) const
        {
            Solver decomp;
            decomp.compute(A);

            if(!decomp.isPositive())
                throw std::runtime_error("DenseCholeskySolver: matrix is not positive semi-definite");

            result = decomp.solve(b);
        }
    };

    /** Base class for least squares algorithms.
      * It implements the whole optimization strategy except the step
      * calculation. Cannot be instantiated. */
    template<typename Scalar,
        typename ErrorFunction,
        typename StepSize,
        typename Callback,
        typename FiniteDifferences>
    class LeastSquaresAlgorithm
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    protected:
        ErrorFunction errorFunction_;
        StepSize stepSize_;
        Callback callback_;
        FiniteDifferences finiteDifferences_;

        Index maxIt_;
        Scalar minStepLen_;
        Scalar minGradLen_;
        Scalar minError_;
        Index verbosity_;

        virtual void calculateStep(const Vector &xval,
            const Vector &fval,
            const Matrix &jacobian,
            const Vector &gradient,
            Vector &step) = 0;

        void evaluateErrorFunction(const Vector &xval, Vector &fval, Matrix &jacobian)
        {
            jacobian.resize(0, 0);
            errorFunction_(xval, fval, jacobian);
            if(jacobian.size() == 0)
                finiteDifferences_(xval, fval, jacobian);
        }

        std::string vector2str(const Vector &vec) const
        {
            std::stringstream ss1;
            ss1 << std::fixed << std::showpoint << std::setprecision(6);
            std::stringstream ss2;
            ss2 << '[';
            for(Index i = 0; i < vec.size(); ++i)
            {
                ss1 << vec(i);
                ss2 << std::setfill(' ') << std::setw(10) << ss1.str();
                if(i != vec.size() - 1)
                    ss2 << ' ';
                ss1.str("");
            }
            ss2 << ']';

            return ss2.str();
        }

    public:
        struct Result
        {
            Vector xval;
            Vector fval;
            Scalar error;
            Index iterations;
            bool converged;
        };

        LeastSquaresAlgorithm()
            : errorFunction_(), stepSize_(), callback_(), finiteDifferences_(),
            maxIt_(0), minStepLen_(1e-9), minGradLen_(1e-9), minError_(0),
            verbosity_(0)
        { }

        virtual ~LeastSquaresAlgorithm()
        { }

        /** Set the number of threads used to compute gradients.
          * This only works if OpenMP is enabled.
          * Set to 0 to allow automatic detection of thread number.
          * @param threads number of threads to be used */
        void setThreads(const Index threads)
        {
            finiteDifferences_.setThreads(threads);
        }

        /** Set the difference for gradient estimation with finite differences.
          * @param eps numerical epsilon */
        void setNumericalEpsilon(const Scalar eps)
        {
            finiteDifferences_.setNumericalEpsilon(eps);
        }

        void setErrorFunction(const ErrorFunction &errorFunction)
        {
            errorFunction_ = errorFunction;
        }

        void setCallback(const Callback &callback)
        {
            callback_ = callback;
        }

        void setStepSize(const StepSize &stepSize)
        {
            stepSize_ = stepSize;
        }

        /** Set the maximum number of iterations.
          * Set to 0 or negative for infinite iterations.
          * @param iterations maximum number of iterations */
        void setMaxIterations(const Index iterations)
        {
            maxIt_ = iterations;
        }

        /** Set the minimum step length between two iterations.
          * If the step length falls below this value, the optimizer stops.
          * @param steplen minimum step length */
        void setMinStepLength(const Scalar steplen)
        {
            minStepLen_ = steplen;
        }

        /** Set the minimum gradient length.
          * If the gradient length falls below this value, the optimizer stops.
          * @param gradlen minimum gradient length */
        void setMinGradientLength(const Scalar gradlen)
        {
            minGradLen_ = gradlen;
        }

        /** Set the minimum squared error.
          * If the error falls below this value, the optimizer stops.
          * @param error minimum error */
        void setMinError(const Scalar error)
        {
            minError_ = error;
        }

        /** Set the level of verbosity to print status information after each
          * iteration.
          * Set to 0 to deacticate any output.
          * @param verbosity level of verbosity */
        void setVerbosity(const Index verbosity)
        {
            verbosity_ = verbosity;
        }

        Result minimize(const Vector &initialGuess)
        {
            finiteDifferences_.setErrorFunction(
                [this](const Vector &xval, Vector &fval)
                { Matrix tmp; this->errorFunction_(xval, fval, tmp); });
            stepSize_.setErrorFunction(
                [this](const Vector &xval, Vector &fval, Matrix &jacobian)
                { this->evaluateErrorFunction(xval, fval, jacobian); });

            Vector xval = initialGuess;
            Vector fval;
            Matrix jacobian;
            Vector gradient;
            Scalar gradLen = minGradLen_ + 1;
            Scalar stepSize;
            Scalar error = minError_ + 1;
            Vector step = Vector::Zero(xval.size());
            Scalar stepLen = minStepLen_ + 1;
            bool callbackResult = true;

            Index iterations = 0;
            while((maxIt_ <= 0 || iterations < maxIt_) &&
                gradLen >= minGradLen_ &&
                stepLen >= minStepLen_ &&
                error >= minError_ &&
                callbackResult)
            {
                xval -= step;
                evaluateErrorFunction(xval, fval, jacobian);
                error = 0.5 * fval.squaredNorm();
                gradient = jacobian.transpose() * fval;
                gradLen = gradient.norm();

                calculateStep(xval, fval, jacobian, gradient, step);

                // update step according to step size
                stepSize = stepSize_(xval, fval, jacobian, gradient, step);
                step *= stepSize;
                stepLen = step.norm();
                // evaluate callback an save its result
                callbackResult = callback_(iterations, xval, fval, jacobian,
                    gradient, step);

                if(verbosity_ > 0)
                {
                    std::stringstream ss;
                    ss << "it=" << std::setfill('0')
                        << std::setw(4) << iterations
                        << std::fixed << std::showpoint << std::setprecision(6)
                        << "    stepsize=" << stepSize
                        << "    steplen=" << stepLen
                        << "    gradlen=" << gradLen;

                    if(verbosity_ > 2)
                        ss << "    callback=" << (callbackResult ? "true" : "false");

                    ss << "    error=" << error;
                    ss << "    fval=" << vector2str(fval);

                    if(verbosity_ > 1)
                        ss << "    xval=" << vector2str(xval);
                    if(verbosity_ > 3)
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
            result.converged = stepLen < minStepLen_ ||
                gradLen < minGradLen_ ||
                error < minError_;

            return result;
        }

    };

    template<typename Scalar,
        typename ErrorFunction,
        typename StepSize=BarzilaiBorwein<Scalar>,
        typename Callback=NoCallback<Scalar>,
        typename FiniteDifferences=CentralDifferences<Scalar>>
    class GradientDescent : public LeastSquaresAlgorithm<Scalar, ErrorFunction,
        StepSize, Callback, FiniteDifferences>
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    protected:
        void calculateStep(const Vector &,
            const Vector &,
            const Matrix &,
            const Vector &gradient,
            Vector &step) override
        {
            step = gradient;
        }
    public:
        GradientDescent()
            : LeastSquaresAlgorithm<Scalar, ErrorFunction,
                StepSize, Callback, FiniteDifferences>()
        { }

    };

    template<typename Scalar,
        typename ErrorFunction,
        typename StepSize=ArmijoBacktracking<Scalar>,
        typename Callback=NoCallback<Scalar>,
        typename FiniteDifferences=CentralDifferences<Scalar>,
        typename Solver=DenseSVDSolver<Scalar>>
    class GaussNewton : public LeastSquaresAlgorithm<Scalar, ErrorFunction,
        StepSize, Callback, FiniteDifferences>
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    protected:
        void calculateStep(const Vector &,
            const Vector &,
            const Matrix &jacobian,
            const Vector &gradient,
            Vector &step) override
        {
            Solver solver;

            Matrix A = jacobian.transpose() * jacobian;
            solver(A, gradient, step);
        }
    public:
        GaussNewton()
            : LeastSquaresAlgorithm<Scalar, ErrorFunction,
                StepSize, Callback, FiniteDifferences>()
        { }
    };

    template<typename Scalar,
        typename ErrorFunction,
        typename StepSize=ConstantStepSize<Scalar>,
        typename Callback=NoCallback<Scalar>,
        typename FiniteDifferences=CentralDifferences<Scalar>,
        typename Solver=DenseSVDSolver<Scalar>>
    class LevenbergMarquardt : public LeastSquaresAlgorithm<Scalar, ErrorFunction,
        StepSize, Callback, FiniteDifferences>
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    protected:
        Scalar increase_;
        Scalar decrease_;
        Scalar lambda_;
        Index maxItLM_;

        void calculateStep(const Vector &xval,
            const Vector &fval,
            const Matrix &jacobian,
            const Vector &gradient,
            Vector &step) override
        {
            Solver solver;
            Scalar error = 0.5 * fval.squaredNorm();
            Scalar errorN = error + 1;

            Vector xvalN;
            Vector fvalN;
            Matrix jacobianN;

            Matrix jacobianSq = jacobian.transpose() * jacobian;
            Matrix A;

            Index iterations = 0;
            while((maxItLM_ <= 0 || iterations < maxItLM_) &&
                errorN > error)
            {
                A = jacobianSq;
                // add identity matrix
                for(Index i = 0; i < A.rows(); ++i)
                    A(i, i) += lambda_;

                solver(A, gradient, step);

                xvalN = xval - step;
                this->errorFunction_(xvalN, fvalN, jacobianN);
                errorN = 0.5 * fvalN.squaredNorm();

                if(errorN > error)
                    lambda_ *= increase_;
                else
                    lambda_ *= decrease_;

                ++iterations;
            }
        }
    public:
        LevenbergMarquardt()
            : LeastSquaresAlgorithm<Scalar, ErrorFunction,
                StepSize, Callback, FiniteDifferences>(), increase_(2),
                decrease_(0.5), lambda_(1), maxItLM_(0)
        { }

        /** Set the initial gradient descent factor of levenberg marquardt.
          * @param lambda gradient descent factor */
        void setLambda(const Scalar lambda)
        {
            lambda_ = lambda;
        }

        /** Set maximum iterations of the levenberg marquardt optimization.
          * Set to 0 or negative for infinite iterations.
          * @param iterations maximum iterations for lambda search */
        void setMaxIterationsLM(const Index iterations)
        {
            maxItLM_ = iterations;
        }

        /** Set the increase factor for the lambda damping.
          * Make sure the value is greater than 1.
          * @param increase factor for increasing lambda */
        void setLambdaIncrease(const Scalar increase)
        {
            assert(increase > 1);
            increase_ = increase;
        }

        /** Set the decrease factor for the lambda damping.
          * Make sure the value is in (0, 1).
          * @param increase factor for increasing lambda */
        void setLambdaDecrease(const Scalar decrease)
        {
            assert(decrease < 1);
            assert(decrease > 0);
            decrease_ = decrease;
        }

    };
}

#endif
