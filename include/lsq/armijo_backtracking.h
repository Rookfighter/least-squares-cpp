/*
 * armijo_backtracking.h
 *
 *  Created on: 23 May 2018
 *      Author: Fabian Meyer
 */

#ifndef LSQ_ARMIJO_BACKTRACKING_H_
#define LSQ_ARMIJO_BACKTRACKING_H_

#include "lsq/line_search_algorithm.h"

namespace lsq
{
    /** Implementation of the ArmijoBacktracking line search algorithm. */
    class ArmijoBacktracking : public LineSearchAlgorithm
    {
    private:
        double beta_;
        double gamma_;

        // value and jacobian for reference (eval armijo condition)
        Eigen::VectorXd refErrVal_;
        Eigen::MatrixXd refErrJac_;
        // value and jacobian for each calculated step length
        Eigen::VectorXd currErrVal_;
        Eigen::MatrixXd currErrJac_;

        Eigen::VectorXd refGrad_;

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
            : LineSearchAlgorithm(), beta_(0.8), gamma_(0.1)
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

        double search(const Eigen::VectorXd &state,
            const Eigen::VectorXd &step,
            const std::vector<ErrorFunction *> &errFuncs) override
        {
            // start with maximum step length and decrease
            double result = maxStepLen_;

            // calculate error of state without step as reference
            evalErrorFuncs(state, errFuncs, refErrVal_, refErrJac_);
            double refVal = squaredError(refErrVal_);

            // calculate error of current step with full step length
            evalErrorFuncs(state + result * step, errFuncs, currErrVal_, currErrJac_);
            double currVal = squaredError(currErrVal_);

            // reference gradient of target function
            refGrad_ = refErrJac_.transpose() * refErrVal_;

            // ensure step is descent direction
            assert(refGrad_.size() == step.size());
            // assert((refGrad.transpose() * step)(0) < 0);

            size_t iterations = 0;
            // check for armijo condition
            while(
                !armijoCondition(currVal, refVal, refGrad_, step, result, gamma_) &&
                (maxIt_ == 0 || iterations < maxIt_) && result > minStepLen_)
            {
                // decrease step length
                result *= beta_;

                // calculate error of new state
                evalErrorFuncs(state + result * step, errFuncs, currErrVal_, currErrJac_);
                currVal = squaredError(currErrVal_);

                ++iterations;
            }

            // limit step length by minimum step length
            result = std::max(result, minStepLen_);

            return result;
        }
    };
}

#endif
