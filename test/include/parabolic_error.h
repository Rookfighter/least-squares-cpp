
#ifndef LSQCPP_PARABOLIC_ERROR_
#define LSQCPP_PARABOLIC_ERROR_

template<typename Scalar>
struct ParabolicError
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    void operator()(const Vector &xval, Vector &fval, Matrix &jacobian) const
    {
        assert(xval.size() % 2 == 0);

        // calculate the error vector
        fval.resize(xval.size() / 2);
        for(lsq::Index i = 0; i < fval.size(); ++i)
            fval(i) = xval(i*2) * xval(i*2) + xval(i*2+1) * xval(i*2+1);

        // calculate the jacobian explicitly
        jacobian.setZero(fval.size(), xval.size());
        for(lsq::Index i = 0; i < jacobian.rows(); ++i)
        {
            jacobian(i, i*2) = 2* xval(i*2);
            jacobian(i, i*2+1) = 2* xval(i*2+1);
        }
    }
};

template<typename Scalar>
struct ParabolicErrorNoJacobian
{
    ParabolicError<Scalar> error_;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    void operator()(const Vector &xval, Vector &fval) const
    {
        Matrix jac;
        error_(xval, fval, jac);
    }
};

template<typename Scalar>
struct ParabolicErrorInverseJacobian
{
    ParabolicError<Scalar> error_;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    void operator()(const Vector &xval, Vector &fval, Matrix &jacobian)
    {
        error_(xval, fval, jacobian);
        jacobian *= -1;
    }
};

#endif