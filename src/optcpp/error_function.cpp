/*
 * error_function.cpp
 *
 *  Created on: 04 May 2018
 *      Author: Fabian Meyer
 */


#include "optcpp/error_function.h"

namespace opt
{
    double squaredError(const Eigen::VectorXd &errVec)
    {
        return 0.5 * (errVec.transpose() * errVec)(0);
    }

    static size_t dimension(const std::vector<ErrorFunction *> &errFuncs)
    {
        size_t sum = 0;
        for(const ErrorFunction *e : errFuncs)
            sum += e->dimension();
        return sum;
    }

    ErrorFunction::Result evalErrorFuncs(const Eigen::VectorXd &state,
        const std::vector<ErrorFunction*> &errFuncs)
    {
        ErrorFunction::Result result;
        result.val.setZero(opt::dimension(errFuncs));
        result.jac.setZero(opt::dimension(errFuncs), state.size());

        // keep track of the error index since error functions can
        // return arbitrary amount of values
        size_t eidx = 0;
        for(unsigned int i = 0; i < errFuncs.size(); ++i)
        {
            const ErrorFunction *err = errFuncs[i];

            // calculate error function of the current state
            ErrorFunction::Result res = err->eval(state);
            for(unsigned int j = 0; j < res.val.size(); ++j)
                result.val(eidx + j) = res.val(j);

            // copy whole jacobian into one row of coefficient matrix
            for(unsigned int row = 0; row < res.jac.rows(); ++row)
            {
                for(unsigned int col = 0; col < res.jac.cols(); ++col)
                    result.jac(eidx + row, col) = res.jac(row, col);
            }

            eidx += res.val.size();
        }

        return result;
    }
}
