/* lsqcpp.h
 *
 * Author: Fabian Meyer
 * Created On: 22 Jul 2019
 * License: MIT
 */

#ifndef LSQCPP_LSQCPP_H_
#define LSQCPP_LSQCPP_H_

#include <Eigen/Geometry>
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
        typedef std::function<void(const Vector &, Vector &)> Objective;
    private:
        Scalar eps_;
        Index threads_;
        Objective objective_;
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

        void setObjective(const Objective &objective)
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
                Vector fvalN
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
        typedef std::function<void(const Vector &, Vector &)> Objective;
    private:
        Scalar eps_;
        Index threads_;
        Objective objective_;
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

        void setObjective(const Objective &objective)
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
        typedef std::function<Scalar(const Vector &)> Objective;
    private:
        Scalar eps_;
        Index threads_;
        Objective objective_;
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

        void setObjective(const Objective &objective)
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
            for(size_t i = 0; i < fvalN.size(); ++i)
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
        typedef std::function<Scalar(const Vector &, Vector &)> Objective;
    private:
        Scalar stepSize_;
    public:

        ConstantStepSize()
            : ConstantStepSize(0.7)
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

        void setObjective(const Objective &)
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
        typedef std::function<void(const Vector &, Vector &, Matrix &)> Objective;
    private:
        Scalar decrease_;
        Scalar c1_;
        Scalar c2_;
        Scalar minStep_;
        Scalar maxStep_;
        Index maxIt_;
        Objective objective_;

    public:
        WolfeBacktracking()
            : WolfeBacktracking(0.8, 1e-4, 0.9, 1e-10, 1.0, 0)
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

        void setObjective(const Objective &objective)
        {
            objective_ = objective;
        }

        Scalar operator()(const Vector &xval,
            const Vector &fval,
            const Matrix &jacobian,
            const Vector &gradient,
            const Vector &step)
        {
            assert(objective_);

            Scalar stepSize = maxStep_ / decrease_;
            Matrix jacobianN;
            Vector gradientN;
            Vector xvalN;
            Vector fvalN;
            Scalar fvalNNorm;

            Scalar error = fval.squaredNorm();
            Scalar stepGrad = step.dot(gradient);
            bool armijoCondition = false;
            bool wolfeCondition = false;

            Index iterations = 0;
            while((maxIt_ <= 0 || iterations < maxIt_) &&
                stepSize * decrease_ >= minStep_ &&
                !(armijoCondition && wolfeCondition))
            {
                stepSize = decrease_ * stepSize;
                xvalN = xval - stepSize * gradient;
                objective_(xvalN, fvalN, jacobianN);
                errorN = fvalN.squaredNorm();
                gradientN = jacobianN.transpose() * fvalN;

                armijoCondition = errorN <= error + c1_ * stepSize * stepGrad;
                wolfeCondition = step.dot(gradientN) >= c2_ * stepGrad;

                ++iterations;
            }

            return stepSize;
        }
    };

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
        Objective objective_;
        StepSize stepSize_;
        Callback callback_;
        FiniteDifferences finiteDifferences_;

        Index maxIt_;
        Scalar minStepLen_;
        Index verbosity_;

        virtual void calculateStep(const Vector &xval,
            const Vector &fval,
            const Matrix &jacobian,
            const Vector &gradient,
            Vector &step) = 0;

        void evaluateObjective(const Vector &xval, Vector &fval, Matrix &jacobian)
        {
            jacobian.resize(0, 0);
            objective_(xval, fval, jacobian);
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
            Index iterations;
            bool converged;
        };

        LeastSquaresAlgorithm()
            : objective_(), stepSize_(), callback_(), finiteDifferences_(),
            maxIt_(0), minStepLen_(1e-6), verbosity_(0)
        { }

        virtual ~LeastSquaresAlgorithm()
        { }

        void setThreads(const Index threads)
        {
            finiteDifferences_.setThreads(threads);
        }

        void setNumericalEpsilon(const Scalar eps)
        {
            finiteDifferences_.setNumericalEpsilon(eps);
        }

        void setObjective(const Objective &objective)
        {
            objective_ = objective;
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

        void setVerbosity(const Index verbosity)
        {
            verbosity_ = verbosity;
        }

        Result minimize(const Vector &initialGuess)
        {
            finiteDifferences_.setObjective(
                [this](const Vector &xva, Vector &fval)
                { Matrix tmp; this->objective_(xval, fval, tmp); });
            stepSize_.setObjective(
                [this](const Vector &xval, Vector &fval, Matrix &jacobian)
                { evaluateObjective(xval, fval, jacobian); });

            Vector xval = initialGuess;
            Vector fval;
            Matrix jacobian;
            Vector gradient;
            Scalar stepSize;
            Scalar error = minError_ + 1;
            Vector step = Vector::Zero(xval.size());
            Scalar stepLen = minStepLen_ + 1;
            bool callbackResult = true;

            Index iterations = 0;
            while((maxIt_ <= 0 || iterations < maxIt_) &&
                stepLen >= minStepLen_ &&
                error >= minError_ &&
                callbackResult)
            {
                xval -= step;
                evaluateObjective(xval, fval, jacobian);
                error = 0.5 * fval.squaredNorm();
                gradient = jacobian.transpose() * fval;

                calculateStep(xval, fval, jacobian, gradient, step);

                // update step according to step size and momentum
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
                        << "    steplen=" << stepLen;

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
            result.iterations = iterations;
            result.converged = stepLen < minStepLen_;

            return result;
        }

    };

    template<typename Scalar,
        typename ErrorFunction,
        typename StepSize=WolfeBacktracking<Scalar>,
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
            step = -gradient;
        }
    public:
        GradientDescent()
            : LeastSquaresAlgorithm<Scalar, ErrorFunction,
                StepSize, Callback, FiniteDifferences>()
        { }

    };

    template<typename Scalar,
        typename ErrorFunction,
        typename StepSize=WolfeBacktracking<Scalar>,
        typename Callback=NoCallback<Scalar>,
        typename FiniteDifferences=CentralDifferences<Scalar>>
    class GaussNewton : public LeastSquaresAlgorithm<Scalar, ErrorFunction,
        StepSize, Callback, FiniteDifferences>
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::JacobiSVD<Matrix, Eigen::FullPivHouseholderQRPreconditioner>
            SVDSolver;
    protected:
        Matrix A_;

        void calculateStep(const Vector &,
            const Vector &,
            const Matrix &jacobian,
            const Vector &gradient
            Vector &step) override
        {
            A_ = jacobian.transpose() * jacobian;

            SVDSolver solver(A_, Eigen::ComputeFullU | Eigen::ComputeFullV);
            step = -solver.solve(gradient);
        }
    public:
        GaussNewton()
            : LeastSquaresAlgorithm<Scalar, ErrorFunction,
                StepSize, Callback, FiniteDifferences>()
        { }
    };

    template<typename Scalar,
        typename ErrorFunction,
        typename StepSize=WolfeBacktracking<Scalar>,
        typename Callback=NoCallback<Scalar>,
        typename FiniteDifferences=CentralDifferences<Scalar>>
    class LevenbergMarquardt : public LeastSquaresAlgorithm<Scalar, ErrorFunction,
        StepSize, Callback, FiniteDifferences>
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::JacobiSVD<Matrix, Eigen::FullPivHouseholderQRPreconditioner>
            SVDSolver;
    protected:
        Matrix A_;

        void calculateStep(const Vector &xval,
            const Vector &fval,
            const Matrix &jacobian,
            const Vector &gradient
            Vector &step) override
        {
            Scalar error = fval.squaredNorm();
            A_ = jacobian.transpose() * jacobian;

            SVDSolver solver(A_, Eigen::ComputeFullU | Eigen::ComputeFullV);
            step = -solver.solve(gradient);
        }
    public:
        LevenbergMarquardt()
            : LeastSquaresAlgorithm<Scalar, ErrorFunction,
                StepSize, Callback, FiniteDifferences>()
        { }

    };
}

#endif
