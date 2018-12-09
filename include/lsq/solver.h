/*
 * solver.h
 *
 *  Created on: 11 Oct 2018
 *      Author: Fabian Meyer
 */

#ifndef LSQ_SOLVER_H_
#define LSQ_SOLVER_H_

#include "lsq/linear_equation_system.h"

namespace lsq
{
    template<typename Scalar>
    class Solver
    {
    public:
        Solver()
        {}
        virtual ~Solver()
        {}

        virtual void solve(const LinearEquationSystem<Scalar> &system,
            Vector<Scalar> &result) const = 0;
    };
}

#endif
