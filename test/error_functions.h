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

    void eval(const Eigen::VectorXd &state,
        Eigen::VectorXd &outValue,
        Eigen::MatrixXd &outJacobian) const override
    {
        // assert(factors.size == state.size())
        outValue = factors.transpose() * state;
        outJacobian = factors.transpose();
    }
};

class LinearErrFuncNoJac : public opt::ErrorFunction
{
public:
    Eigen::VectorXd factors;

    size_t dimension() const override
    {
        return 1;
    }

    void eval(const Eigen::VectorXd &state,
        Eigen::VectorXd &outValue,
        Eigen::MatrixXd &) const override
    {
        // assert(factors.size == state.size())
        outValue = factors.transpose() * state;
    }
};

#endif
