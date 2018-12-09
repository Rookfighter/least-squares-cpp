/*
 * error_function.h
 *
 *  Created on: 04 May 2018
 *      Author: Fabian Meyer
 */

#ifndef LSQ_ERROR_FUNCTION_H_
#define LSQ_ERROR_FUNCTION_H_

#include "lsq/matrix.h"

namespace lsq
{
    /** Interface to define error functions for optimization problems. */
    template<typename Scalar, typename Error>
    class ErrorFunction
    {
    private:
        Scalar fdEps_;

    public:
        ErrorFunction()
            : fdEps_(std::sqrt(std::numeric_limits<double>::epsilon()))
        {}

        virtual ~ErrorFunction()
        {}

        void setNumericalEps(const Scalar eps)
        {
            fdEps_ = eps;
        }

        /** Internal evaluation of the error function and its jacobian.
-         *  If the function calculates no jacobian then finite differences
-         *  is used to approximate it.
-         *  @param state current state estimate
-         *  @param outValue function value of the error function
-         *  @param outJacobian jacobian of the error function */
-        virtual void _evaluate(const Eigen::VectorXd &state,
-            Eigen::VectorXd &outValue,
-            Eigen::MatrixXd &outJacobian) const = 0;

        void computeFiniteDifferences(const Vector<Scalar> &state,
            const Vector<Scalar> &errValue,
            Matrix<Scalar> &outJacobian)
        {
            Eigen::VectorXd stateTmp;
            Eigen::VectorXd errValueTmp;
            Eigen::MatrixXd errJacobianTmp;

            outJacobian.resize(errValue.size(), state.size());

            for(unsigned int i = 0; i < state.size(); ++i)
            {
                stateTmp = state;
                stateTmp(i) += fdEps_;

                errFunctor_(stateTmp, errValueTmp, errJacobianTmp);
                assert(errValueTmp.size() == errValue.size());

                outJacobian.col(i) = (errValueTmp - errValue) / diff;
            }
        }

        /** Evaluates the error function and its jacobian.
         *  @param state current state estimate
         *  @param outValue function value of the error function
         *  @param outJacobian jacobian of the error function */
        void evaluate(const Vector<Scalar> &state,
            Vector<Scalar> &outValue,
            Matrix<Scalar> &outJacobian)
        {
            outJacobian.resize(0, 0);
            errFunctor_(state, outValue, outJacobian);

            // if no jacobian was computed use finite differences
            if(outJacobian.size() == 0)
                computeFiniteDifferences(state, outValue, outJacobian);

            assert(outJacobian.rows() == outValue.size());
            assert(outJacobian.cols() == state.size());
        }
    };

    /** Calculates the squared error of a least squares problem given the error
     *  vector. Calculates as:  0.5 * err^T * err
     *  @param errorVec vector of error values
     *  @return squared error */
    template<typename Scalar>
    inline Scalar squaredError(const Vector<Scalar> &errorVec)
    {
        return Scalar(0.5) * errorVec.squaredNorm();
    }
}

#endif
