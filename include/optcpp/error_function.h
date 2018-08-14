/*
 * error_function.h
 *
 *  Created on: 04 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_ERROR_FUNCTION_H_
#define OPT_ERROR_FUNCTION_H_

#include <Eigen/Dense>

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

        /** Returns the length of the output vector of the error function.
         *  This is used for prediction the length of the final function vetcor
         *  @return length of the result vector */
        virtual size_t dimension() const = 0;

        /** Evaluates the error function and its jacobian.
         *  @param state current state estimate
         *  @param outValue function value of the error function
         *  @param outJacobian jacobian of the error function */
        virtual void eval(const Eigen::VectorXd &state,
            Eigen::VectorXd &outValue,
            Eigen::MatrixXd &outJacobian) const = 0;
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

        Eigen::VectorXd errVal;
        Eigen::MatrixXd errJac;

        // keep track of the error index since error functions can
        // return arbitrary amount of values
        size_t eidx = 0;
        for(unsigned int i = 0; i < errFuncs.size(); ++i)
        {
            const ErrorFunction *err = errFuncs[i];

            // calculate error function of the current state
            err->eval(state, errVal, errJac);
            for(unsigned int j = 0; j < errVal.size(); ++j)
                outValue(eidx + j) = errVal(j);

            // copy whole jacobian into one row of coefficient matrix
            for(unsigned int row = 0; row < errJac.rows(); ++row)
            {
                for(unsigned int col = 0; col < errJac.cols(); ++col)
                    outJacobian(eidx + row, col) = errJac(row, col);
            }

            eidx += errVal.size();
        }
    }
}

#endif
