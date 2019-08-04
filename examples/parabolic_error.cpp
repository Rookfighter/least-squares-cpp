/* parabolic_error.cpp
 *
 *  Created on: 11 Dec 2018
 *      Author: Fabian Meyer
 */

#include <lsqcpp.h>

struct ParabolicError
{
    Eigen::VectorXd values;

    void operator()(const Eigen::VectorXd &xval,
        Eigen::VectorXd &fval,
        Eigen::MatrixXd &) const
    {
        fval.resize(xval.size() / 2);
        for(lsq::Index i = 0; i < fval.size(); ++i)
            fval(i) = xval(i*2) * xval(i*2) + xval(i*2+1) * xval(i*2+1);
        fval -= values;
    }
};

int main()
{
    lsq::GaussNewton<double, ParabolicError, lsq::WolfeBacktracking<double>> optimizer;

    return 0;
}
