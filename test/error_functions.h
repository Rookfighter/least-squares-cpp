/*
 * error_functions.h
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_TEST_ERROR_FUNCTIONS_H_
#define OPT_TEST_ERROR_FUNCTIONS_H_

#include <optcpp/error_function.h>

class LinearErrFunc : public opt::ErrorFunction
{
public:
    Eigen::VectorXd factors;

    size_t dimension() const override
    {
        return 1;
    }

    Result eval(const Eigen::VectorXd &state) const override
    {
        // assert(factors.size == state.size())

        Result result;
        result.val = factors.transpose() * state;
        result.jac = factors.transpose();

        return result;
    }
};

#endif
