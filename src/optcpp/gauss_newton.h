/*
 * gauss_newton.h
 *
 *  Created on: 03 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_GAUSS_NEWTON_H_
#define OPT_GAUSS_NEWTON_H_

#include "optcpp/optimization_algorithm.h"

namespace opt
{
    /** Implementation of the gauss newton optimization algorithm. */
    class GaussNewton : public OptimizationAlgorithm
    {
      private:
        double damping_;

      public:
        GaussNewton();
        GaussNewton(const GaussNewton &gn) = delete;
        ~GaussNewton();

        void setDamping(const double damping);

        Eigen::VectorXd calcStepUpdate(const Eigen::VectorXd &state);
    };
}

#endif
