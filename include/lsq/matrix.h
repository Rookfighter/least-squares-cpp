/*
 * matrix.h
 *
 *  Created on: 09 Dec 2018
 *      Author: Fabian Meyer
 */

#ifndef LSQ_MATRIX_H_
#define LSQ_MATRIX_H_

#include <Eigen/Geometry>

namespace lsq
{

    template<typename Scalar>
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    template<typename Scalar>
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    typedef Vector<double> Vectord;
    typedef Vector<float> Vectorf;

    typedef Matrix<double> Matrixd;
    typedef Matrix<float> Matrixf;
}

#endif
