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
    class GradientDescent : public OptimizationAlgorithm
    {
    private:
        double stepWidth(const Eigen::VectorXd &state,
                         const Eigen::MatrixXd &jac) const;
    public:
        GradientDescent();
        ~GradientDescent();

        Eigen::VectorXd calcStepUpdate(const Eigen::VectorXd &state) override;
    };
}

#endif
