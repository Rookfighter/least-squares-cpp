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

        struct Result
        {
            /** Function value. */
            Eigen::VectorXd val;
            /** Function jacobian. */
            Eigen::MatrixXd jac;
        };

        /** Returns the length of the output vector of the error function.
         * This is used for prediction the length of the final function vetcor
         * @return length of the result vector */
        virtual size_t dimension() const = 0;

        /**
         * Evaluates the error function and its jacobian.
         * @return struct containing function value and jacobian */
        virtual Result eval(const Eigen::VectorXd &state) const = 0;
    };

    /** Calculates the squared error of a least squares problem given the error
     *  vector. Calculates as:  0.5 * err^T * err
     *  @param errVec vector of error values
     *  @return squared error */
    double squaredError(const Eigen::VectorXd &errVec);

    /** Calculates the value and jacobians of a vector of error functions.
     *  @param state current state vector
     *  @param errFuncs vector of error functions
     *  @return error vector and jacobian */
    ErrorFunction::Result evalErrorFuncs(const Eigen::VectorXd &state,
        const std::vector<ErrorFunction *> &errFuncs);
}

#endif
