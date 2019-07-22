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
            const Matrix &jacobian)
        {
            assert(objective_);

            Scalar stepSize = maxStep_ / decrease_;
            Matrix jacobianN;
            Vector gradientN;
            Vector xvalN;
            Vector fvalN;
            Scalar fvalNNorm;

            Scalar fvalNorm = fval.squaredNorm();
            Vector gradient = jacobian.transpose() * fval;
            Scalar gradientDot = gradient.dot(gradient);
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
                fvalNNorm = fvalN.squaredNorm();
                gradientN = jacobianN.transpose() * fvalN;

                armijoCondition = fvalNNorm <= fvalNorm - c1_ * stepSize * gradientDot;
                wolfeCondition = gradient.dot(gradientN) <= c2_ * gradientDot;

                ++iterations;
            }

            return stepSize;
        }
    };

    template<typename Scalar>
    class GradientDescent
    {

    };

    template<typename Scalar>
    class GaussNewton
    {

    };

    template<typename Scalar>
    class LevenbergMarquardt
    {

    };
}

#endif
