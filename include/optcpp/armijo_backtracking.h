/*
 * armijo_backtracking.h
 *
 *  Created on: 23 May 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_ARMIJO_BACKTRACKING_H_
#define OPT_ARMIJO_BACKTRACKING_H_

#include "optcpp/line_search_algorithm.h"

namespace opt
{
    class ArmijoBacktracking : public LineSearchAlgorithm
    {
    private:
        /** Reduction factor during step calculation. Interval (0,1).
          * Choose not too small, e.g. 0.8. */
        double beta_;

        /** Relaxation of the gradient during step calculation. Interval
          * (0, 0.5). Choose not too big, e.g. 0.1. */
        double gamma_;

        /** Maximum allowed step length. Typically 1.0. */
        double maxStepLen_;

    public:
        ArmijoBacktracking();
        ~ArmijoBacktracking();

        void setBeta(const double beta);
        void setGamma(const double gamma);
        void setMaxStepLen(const double stepLen);

        double calcStepLength(
            const Eigen::VectorXd &state,
            const Eigen::VectorXd &step,
            const std::vector<Constraint *> &constraints) const override;
    };
}

#endif
