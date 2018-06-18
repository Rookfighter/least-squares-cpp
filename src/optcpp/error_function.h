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
        ErrorFunction() { }
        virtual ~ErrorFunction() { }

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
}

#endif
