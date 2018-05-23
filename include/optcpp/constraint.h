/*
 * constraint.h
 *
 *  Created on: 04 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_CONSTRAINT_H_
#define OPT_CONSTRAINT_H_

#include <Eigen/Dense>

namespace opt
{
    /**
     * Interface to define optimization constraints.
     */
    class Constraint
    {
    public:
        Constraint() { }
        virtual ~Constraint() { }

        struct Result
        {
            /** Function value. */
            Eigen::VectorXd val;
            /** Function jacobian. */
            Eigen::MatrixXd jac;
        };

        virtual unsigned int outputSize() const = 0;

        /**
         * Calculates the error function of the constraint and its jacobian.
         * @return struct containing function value and jacobian
         */
        virtual Result errorFunc(const Eigen::VectorXd &state) const = 0;
    };
}

#endif
