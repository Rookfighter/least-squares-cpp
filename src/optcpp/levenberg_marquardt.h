/*
 * levenberg_marquardt.h
 *
 *  Created on: 09 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_LEVENBERG_MARQUARDT_H_
#define OPT_LEVENBERG_MARQUARDT_H_

#include "optcpp/optimization_algorithm.h"

namespace opt
{

    /** Implementation of the levelberg marquardt optimization algorithm. */
    class LevenbergMarquardt : public OptimizationAlgorithm
    {
    private:
        double damping_;
        double lambda_;
        size_t maxIt_;
    public:
        LevenbergMarquardt();
        ~LevenbergMarquardt();

        void setDamping(const double damping);

        /** Sets the gradient descent factor of levenberg marquardt.
         *  @param lambda gradient descent factor */
        void setLambda(const double lambda);

        /** Sets maximum iterations of the levenberg marquardt optimization.
         *  Set to 0 for infinite iterations.
         *  @param maxIt maximum iteration for optimization */
        void setMaxIterations(const size_t maxIt);

        Eigen::VectorXd calcStepUpdate(const Eigen::VectorXd &state);
    };
}

#endif
