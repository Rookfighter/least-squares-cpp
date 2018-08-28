/*
 * error_function.h
 *
 *  Created on: 04 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_ERROR_FUNCTION_H_
#define OPT_ERROR_FUNCTION_H_

#include <Eigen/Geometry>

namespace opt
{
    /** Interface to define error functions for optimization problems. */
    class ErrorFunction
    {
    public:
        ErrorFunction()
        {}
        virtual ~ErrorFunction()
        {}

        void computeFiniteDifferences(const Eigen::VectorXd &state,
            const Eigen::VectorXd &errValue,
            Eigen::MatrixXd &outJacobian,
            const double diff) const
        {
            Eigen::VectorXd stateTmp;
            Eigen::VectorXd errValueTmp;
            Eigen::MatrixXd errJacobianTmp;

            outJacobian.resize(errValue.size(), state.size());

            for(unsigned int i = 0; i < state.size(); ++i)
            {
                stateTmp = state;
                stateTmp(i) += diff;

                _evaluate(stateTmp, errValueTmp, errJacobianTmp);
                assert(errValueTmp.size() == errValue.size());

                outJacobian.col(i) = (errValueTmp - errValue) / diff;
            }
        }

        /** Returns the length of the output vector of the error function.
         *  This is used for prediction the length of the final function vetcor
         *  @return length of the result vector */
        virtual size_t dimension() const = 0;

        /** Internal evaluation of the error function and its jacobian.
         *  If the function calculates no jacobian then finite differences
         *  is used to approximate it.
         *  @param state current state estimate
         *  @param outValue function value of the error function
         *  @param outJacobian jacobian of the error function */
        virtual void _evaluate(const Eigen::VectorXd &state,
            Eigen::VectorXd &outValue,
            Eigen::MatrixXd &outJacobian) const = 0;

        /** Evaluates the error function and its jacobian.
         *  @param state current state estimate
         *  @param outValue function value of the error function
         *  @param outJacobian jacobian of the error function */
        void evaluate(const Eigen::VectorXd &state,
            Eigen::VectorXd &outValue,
            Eigen::MatrixXd &outJacobian) const
        {
            static const double diff = std::sqrt(
                std::numeric_limits<double>::epsilon());

            outJacobian.resize(0, 0);
            _evaluate(state, outValue, outJacobian);

            // if no jacobian was computed use finite differences
            if(outJacobian.size() == 0)
                computeFiniteDifferences(state, outValue, outJacobian, diff);

            assert(static_cast<size_t>(outValue.size()) == dimension());
            assert(outJacobian.rows() == outValue.size());
            assert(outJacobian.cols() == state.size());
        }
    };

    /** Calculates the squared error of a least squares problem given the error
     *  vector. Calculates as:  0.5 * err^T * err
     *  @param errorVec vector of error values
     *  @return squared error */
    inline double squaredError(const Eigen::VectorXd &errorVec)
    {
        return 0.5 * (errorVec.transpose() * errorVec)(0);
    }

    inline size_t totalDimension(const std::vector<ErrorFunction *> &errFuncs)
    {
        size_t sum = 0;
        for(const ErrorFunction *e : errFuncs)
            sum += e->dimension();
        return sum;
    }

    /** Calculates the value and jacobians of a vector of error functions.
     *  @param state current state vector
     *  @param errFuncs vector of error functions
     *  @param outValue function value of the error function
     *  @param outJacobian jacobian of the error function */
    inline void evalErrorFuncs(const Eigen::VectorXd &state,
        const std::vector<ErrorFunction *> &errFuncs,
        Eigen::VectorXd &outValue,
        Eigen::MatrixXd &outJacobian)
    {
        size_t dim = totalDimension(errFuncs);
        outValue.resize(dim);
        outJacobian.resize(dim, state.size());
        std::vector<size_t> eidx(errFuncs.size());

        Eigen::VectorXd errVal;
        Eigen::MatrixXd errJac;
        // keep track of the error index since error functions can
        // return arbitrary amount of values
        size_t eidx = 0;
        for(unsigned int i = 0; i < errFuncs.size(); ++i)
        {
            const ErrorFunction *errfun = errFuncs[i];

            // calculate error function of the current state
            errfun->evaluate(state, errVal, errJac);

            for(unsigned int j = 0; j < errVal.size(); ++j)
                outValue(eidx + j) = errVal(j);

            // copy whole jacobian into one row of coefficient matrix
            for(unsigned int row = 0; row < errJac.rows(); ++row)
            {
                for(unsigned int col = 0; col < errJac.cols(); ++col)
                    outJacobian(eidx + row, col) = errJac(row, col);
            }

            eidx += errfun->dimension();
        }
    }
}

#endif
