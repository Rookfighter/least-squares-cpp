/*
 * error_functions.h
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_TEST_ERROR_FUNCTIONS_H_
#define OPT_TEST_ERROR_FUNCTIONS_H_

#include <lsq/error_function.h>

template<typename Scalar>
class LinearErrFunc : public lsq::ErrorFunction<Scalar>
{
public:
    lsq::Matrix<Scalar> factors;

    void _evaluate(const lsq::Vector<Scalar> &state,
        lsq::Vector<Scalar> &outValue,
        lsq::Matrix<Scalar> &outJacobian) override
    {
        // assert(factors.size == state.size())
        outValue = factors.transpose() * state;
        outJacobian = factors.transpose();
    }
};

typedef LinearErrFunc<double> LinearErrFuncd;

template<typename Scalar>
class LinearErrFuncNoJac : public lsq::ErrorFunction<Scalar>
{
public:
    lsq::Matrix<Scalar> factors;

    void _evaluate(const lsq::Vector<Scalar> &state,
        lsq::Vector<Scalar> &outValue,
        lsq::Matrix<Scalar> &) override
    {
        // assert(factors.size == state.size())
        outValue = factors.transpose() * state;
    }
};

typedef LinearErrFuncNoJac<double> LinearErrFuncNoJacd;

#endif
