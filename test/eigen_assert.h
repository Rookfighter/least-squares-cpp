/*
 * eigen_assert.h
 *
 *  Created on: 18 Jun 2018
 *      Author: Fabian Meyer
 */

#ifndef OPT_TEST_EIGEN_ASSERT_H_
#define OPT_TEST_EIGEN_ASSERT_H_

#include <catch.hpp>

#define REQUIRE_MAT(a, b, eps)                                                 \
    {                                                                          \
        REQUIRE(a.cols() == b.cols());                                         \
        REQUIRE(a.rows() == b.rows());                                         \
        for(unsigned int _c = 0; _c < a.cols(); ++_c)                          \
        {                                                                      \
            for(unsigned int _r = 0; _r < a.rows(); ++_r)                      \
                REQUIRE(Approx(a(_r, _c)).margin(eps) == b(_r, _c));           \
        }                                                                      \
    }

#endif
