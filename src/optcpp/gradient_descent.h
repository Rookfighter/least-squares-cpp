/*
 * gradient_descent.h
 *
 *  Created on: 07 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_GRADIENT_DESCENT_H_
#define OPT_GRADIENT_DESCENT_H_

#include "optcpp/optimization_algorithm.h"

namespace opt
{
    /** Implementation of the gradient descent optimization algorithm. */
    class GradientDescent : public OptimizationAlgorithm
    {
    private:
        double damping_;
    public:
        GradientDescent();
        ~GradientDescent();

        void setDamping(const double damping);

        Eigen::VectorXd calcStepUpdate(const Eigen::VectorXd &state) override;
    };
}

#endif
