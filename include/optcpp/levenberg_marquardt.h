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

    class LevenbergMarquardt : public OptimizationAlgorithm
    {
    private:
        double damping_;
        double gradientFac_;
    public:
        LevenbergMarquardt();
        ~LevenbergMarquardt();

        void setDamping(const double damping);
        void setGradientFactor(const double fac);

        Eigen::VectorXd calcStepUpdate(const Eigen::VectorXd &state);
    };
}

#endif
