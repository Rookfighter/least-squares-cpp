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
    /** Implementation of the ArmijoBacktracking line search algorithm. */
    class ArmijoBacktracking : public LineSearchAlgorithm
    {
    private:
        double beta_;
        double gamma_;
        double minStepLen_;
        double maxStepLen_;
        size_t maxIt_;

    public:
        ArmijoBacktracking();
        ~ArmijoBacktracking();

        /** Sets the reduction factor during step calculation.
         *  The value must be in the interval (0, 1). Choose not too small,
         *  e.g. 0.8.
         *  @param beta reduction factor */
        void setBeta(const double beta);

        /** Sets the relaxation factor of the linearization on the armijo
         *  condition. The value must be in the interval (0, 0.5). Choose not
         *  too big, e.g. 0.1.
         *  @param gamma relaxation factor */
        void setGamma(const double gamma);

        /** Sets the bounds for the step length. The step length is then
         *  assured to be in the interval [minLen, maxLen].
         *  @param minLen minimum step length
         *  @param maxLen maximum step length */
        void setBounds(const double minLen, const double maxLen);

        /** Sets maximum iterations for the line search.
         *  Set to 0 for infinite iterations.
         *  @param maxIt maximum iterations */
        void setMaxIterations(const size_t maxIt);

        double stepLength(
            const Eigen::VectorXd &state,
            const Eigen::VectorXd &step,
            const std::vector<ErrorFunction *> &errFuncs) const override;
    };
}

#endif
