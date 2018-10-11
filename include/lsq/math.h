/*
 * math.h
 *
 *  Created on: 04 May 2018
 *      Author: Fabian Meyer
 */

#ifndef LSQ_MATH_H_
#define LSQ_MATH_H_

#include <cmath>

namespace lsq
{
    double pi()
    {
        return std::atan(1.0) * 4.0;
    }

    inline bool equals(double a, double b, double eps)
    {
        return std::abs(a - b) <= eps;
    }

    inline bool iszero(double a, double eps)
    {
        return equals(a, 0.0, eps);
    }

    inline double normalizeAngle(double angle)
    {
        while(angle <= -pi())
            angle += 2 * pi();
        while(angle > pi())
            angle -= 2 * pi();
        return angle;
    }
}

#endif
