/*
 * increasing_line_search.h
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_INCREASING_LINE_SEARCH_H_
#define OPT_INCREASING_LINE_SEARCH_H_

#include "optcpp/line_search_algorithm.h"

namespace opt
{
    /** Implementation of a increasing line search algorithm. */
    class IncreasingLineSearch : public LineSearchAlgorithm
    {
      private:
        double beta_;
        double maxStepLen_;
        double minStepLen_;
        size_t maxIt_;

      public:
        IncreasingLineSearch();
        ~IncreasingLineSearch();

        /** Sets the increasing factor during step calculation. The value must
         *  be in the interval (1 inf). Choose not too big, e.g. 2.0.
         *  @param beta increasing factor */
        void setBeta(const double beta);

        /** Sets the bounds for the step length. The step length is then
         *  assured to be in the interval [minLen, maxLen].
         *  @param minLen minimum step length
         *  @param maxLen maximum step length */
        void setBounds(const double minLen, const double maxLen);

        /** Sets maximum iterations for the line search.
         *  Set to 0 for infinite iterations.
         *  @param maxIt maximum iterations */
        void setMaxIterations(const size_t maxIt);

        double stepLength(const Eigen::VectorXd &state,
            const Eigen::VectorXd &step,
            const std::vector<ErrorFunction *> &errFuncs) const override;
    };
}

#endif
