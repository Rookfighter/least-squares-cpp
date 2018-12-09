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
    template<typename Scalar>
    class ArmijoBacktracking : public LineSearchAlgorithm<Scalar>
    {
    private:
        Scalar beta_;
        Scalar gamma_;

        // value and jacobian for reference (eval armijo condition)
        Vector<Scalar> refErrVal_;
        Matrix<Scalar> refErrJac_;
        // value and jacobian for each calculated step length
        Vector<Scalar> currErrVal_;
        Matrix<Scalar> currErrJac_;

        Vector<Scalar> refGrad_;

        bool armijoCondition(const Scalar currVal,
            const Scalar refVal,
            const Vector<Scalar> &refGrad,
            const Vector<Scalar> &step,
            const Scalar stepLen,
            const Scalar gamma) const
        {
            assert(refGrad.size() == step.size());
            return currVal <=
                   refVal + gamma * stepLen * (refGrad.transpose() * step)(0);
        }

    public:
        ArmijoBacktracking()
            : LineSearchAlgorithm<Scalar>(), beta_(0.8), gamma_(0.1)
        {}
        ~ArmijoBacktracking()
        {}

        /** Sets the reduction factor during step calculation.
         *  The value must be in the interval (0, 1). Choose not too small,
         *  e.g. 0.8.
         *  @param beta reduction factor */
        void setBeta(const Scalar beta)
        {
            assert(beta > 0.0 && beta < 1.0);
            beta_ = beta;
        }

        /** Sets the relaxation factor of the linearization on the armijo
         *  condition. The value must be in the interval (0, 0.5). Choose not
         *  too big, e.g. 0.1.
         *  @param gamma relaxation factor */
        void setGamma(const Scalar gamma)
        {
            assert(gamma > 0.0 && gamma < 0.5);
            gamma_ = gamma;
        }

        double search(const Vector<Scalar> &state,
            const Vector<Scalar> &step,
            ErrorFunction<Scalar> &errFunc) override
        {
            // start with maximum step length and decrease
            Scalar result = this->maxStepLen_;

            // calculate error of state without step as reference
            errFunc.evaluate(state, refErrVal_, refErrJac_);
            Scalar refVal = squaredError<Scalar>(refErrVal_);

            // calculate error of current step with full step length
            errFunc.evaluate(state + result * step, currErrVal_, currErrJac_);
            Scalar currVal = squaredError<Scalar>(currErrVal_);

            // reference gradient of target function
            refGrad_ = refErrJac_.transpose() * refErrVal_;

            // ensure step is descent direction
            assert(refGrad_.size() == step.size());
            // assert((refGrad.transpose() * step)(0) < 0);

            size_t iterations = 0;
            // check for armijo condition
            while(
                !armijoCondition(currVal, refVal, refGrad_, step, result, gamma_) &&
                (this->maxIt_ == 0 || iterations < this->maxIt_)
                && result > this->minStepLen_)
            {
                // decrease step length
                result *= beta_;

                // calculate error of new state
                errFunc.evaluate(state + result * step, currErrVal_, currErrJac_);
                currVal = squaredError<Scalar>(currErrVal_);

                ++iterations;
            }

            // limit step length by minimum step length
            result = std::max<Scalar>(result, this->minStepLen_);

            return result;
        }
    };
}

#endif
