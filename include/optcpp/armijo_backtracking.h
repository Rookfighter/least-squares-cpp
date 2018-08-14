/*
 * armijo_backtracking.h
 *
 *  Created on: 23 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_ARMIJO_BACKTRACKING_H_
#define OPT_ARMIJO_BACKTRACKING_H_

#include "optcpp/line_search_algorithm.h"

namespace opt
{
    /** Implementation of the ArmijoBacktracking line search algorithm. */
    class ArmijoBacktracking : public LineSearchAlgorithm
    {
    private:
        double beta_;
        double gamma_;
        double minStepLen_;
        double maxStepLen_;
        size_t maxIt_;

        bool armijoCondition(const double currVal,
            const double refVal,
            const Eigen::VectorXd &refGrad,
            const Eigen::VectorXd &step,
            const double stepLen,
            const double gamma) const
        {
            assert(refGrad.size() == step.size());
            return currVal <=
                   refVal + gamma * stepLen * (refGrad.transpose() * step)(0);
        }

    public:
        ArmijoBacktracking()
            : LineSearchAlgorithm(), beta_(0.8), gamma_(0.1), minStepLen_(1e-4),
            maxStepLen_(1.0), maxIt_(0)
        {}
        ~ArmijoBacktracking()
        {}

        /** Sets the reduction factor during step calculation.
         *  The value must be in the interval (0, 1). Choose not too small,
         *  e.g. 0.8.
         *  @param beta reduction factor */
        void setBeta(const double beta)
        {
            assert(beta_ > 0.0 && beta_ < 1.0);
            beta_ = beta;
        }

        /** Sets the relaxation factor of the linearization on the armijo
         *  condition. The value must be in the interval (0, 0.5). Choose not
         *  too big, e.g. 0.1.
         *  @param gamma relaxation factor */
        void setGamma(const double gamma)
        {
            assert(gamma > 0.0 && gamma < 0.5);
            gamma_ = gamma;
        }

        /** Sets the bounds for the step length. The step length is then
         *  assured to be in the interval [minLen, maxLen].
         *  @param minLen minimum step length
         *  @param maxLen maximum step length */
        void setBounds(const double minLen, const double maxLen)
        {
            assert(minLen < maxLen);
            maxStepLen_ = maxLen;
            minStepLen_ = minLen;
        }

        /** Sets maximum iterations for the line search.
         *  Set to 0 for infinite iterations.
         *  @param maxIt maximum iterations */
        void setMaxIterations(const size_t maxIt)
        {
            maxIt_ = maxIt;
        }

        double stepLength(const Eigen::VectorXd &state,
            const Eigen::VectorXd &step,
            const std::vector<ErrorFunction *> &errFuncs) const override
        {
            // start with maximum step length and decrease
            double result = maxStepLen_;

            // value and jacobian for reference (eval armijo condition)
            Eigen::VectorXd refErrVal;
            Eigen::MatrixXd refErrJac;
            // value and jacobian for each calculated step length
            Eigen::VectorXd currErrVal;
            Eigen::MatrixXd currErrJac;

            // calculate error of state without step as reference
            evalErrorFuncs(state, errFuncs, refErrVal, refErrJac);
            double refVal = squaredError(refErrVal);

            // calculate error of current step with full step length
            evalErrorFuncs(state + result * step, errFuncs, currErrVal, currErrJac);
            double currVal = squaredError(currErrVal);

            // reference gradient of target function
            Eigen::VectorXd refGrad = refErrJac.transpose() * refErrVal;

            // ensure step is descent direction
            assert(refGrad.size() == step.size());
            // assert((refGrad.transpose() * step)(0) < 0);

            size_t iterations = 0;
            // check for armijo condition
            while(
                !armijoCondition(currVal, refVal, refGrad, step, result, gamma_) &&
                (maxIt_ == 0 || iterations < maxIt_) && result > minStepLen_)
            {
                // decrease step length
                result *= beta_;

                // calculate error of new state
                evalErrorFuncs(state + result * step, errFuncs, currErrVal, currErrJac);
                currVal = squaredError(currErrVal);

                ++iterations;
            }

            // limit step length by minimum step length
            result = std::max(result, minStepLen_);

            return result;
        }
    };
}

#endif
