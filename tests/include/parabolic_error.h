
#ifndef LSQCPP_PARABOLIC_ERROR_
#define LSQCPP_PARABOLIC_ERROR_

struct ParabolicError
{
    constexpr static bool ComputesJacobian = true;

    template<typename Scalar, int Inputs, int Outputs>
    void operator()(const Eigen::Matrix<Scalar, Inputs, 1> &xval,
                    Eigen::Matrix<Scalar, Outputs, 1> &fval,
                    Eigen::Matrix<Scalar, Outputs, Inputs> &jacobian) const
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

struct ParabolicErrorNoJacobian
{
    constexpr static bool ComputesJacobian = false;

    template<typename Scalar, int Inputs, int Outputs>
    void operator()(const Eigen::Matrix<Scalar, Inputs, 1> &xval,
                    Eigen::Matrix<Scalar, Outputs, 1> &fval) const
    {
        Eigen::Matrix<Scalar, Outputs, Inputs> jac;
        ParabolicError error;
        error(xval, fval, jac);
    }
};

struct ParabolicErrorInverseJacobian
{
    constexpr static bool ComputesJacobian = true;

    template<typename Scalar, int Inputs, int Outputs>
    void operator()(const Eigen::Matrix<Scalar, Inputs, 1> &xval,
                    Eigen::Matrix<Scalar, Outputs, 1> &fval,
                    Eigen::Matrix<Scalar, Outputs, Inputs> &jacobian) const
    {
        ParabolicError error;
        error(xval, fval, jacobian);
        jacobian *= -1;
    }
};

#endif