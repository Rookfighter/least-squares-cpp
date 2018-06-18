/*
 * linear_equation_system.cpp
 *
 *  Created on: 04 May 2018
 *      Author: Fabian Meyer
 */

#include "optcpp/linear_equation_system.h"

namespace opt
{
    static size_t equations(const std::vector<ErrorFunction *> &errFuncs)
    {
        size_t sum = 0;
        for(const ErrorFunction *e : errFuncs)
            sum += e->dimension();
        return sum;
    }

    static size_t unknowns(const Eigen::VectorXd &state)
    {
        return state.size();
    }

    LinearEquationSystem::LinearEquationSystem()
    {

    }

    LinearEquationSystem::~LinearEquationSystem()
    {

    }

    void LinearEquationSystem::construct(const Eigen::VectorXd &state,
        const std::vector<ErrorFunction *> &errFuncs)
    {
        b.setZero(opt::equations(errFuncs));
        A.setZero(opt::equations(errFuncs), opt::unknowns(state));

        // keep track of the error index since error functions can
        // return arbitrary amount of values
        size_t eidx = 0;
        for(unsigned int i = 0; i < errFuncs.size(); ++i)
        {
            const ErrorFunction *err = errFuncs[i];

            // calculate error function of the current state
            ErrorFunction::Result res = err->eval(state);
            for(unsigned int j = 0; j < res.val.size(); ++j)
                b(eidx + j) = res.val(j);

            // copy whole jacobian into one row of coefficient matrix
            for(unsigned int row = 0; row < res.jac.rows(); ++row)
            {
                for(unsigned int col = 0; col < res.jac.cols(); ++col)
                    A(eidx + row, col) = res.jac(row, col);
            }

            eidx += res.val.size();
        }
    }

    size_t LinearEquationSystem::equations() const
    {
        return b.size();
    }

    size_t LinearEquationSystem::unknowns() const
    {
        return A.cols();
    }

    bool LinearEquationSystem::undertermined() const
    {
        return unknowns() - equations() > 0;
    }


    Eigen::VectorXd LinearEquationSystem::solveSVD() const
    {
        Eigen::JacobiSVD<Eigen::MatrixXd,
            Eigen::FullPivHouseholderQRPreconditioner>
            decomp(A.transpose() * A,
               Eigen::ComputeFullU | Eigen::ComputeFullV);

        return decomp.solve(A.transpose() * b);
    }
}
