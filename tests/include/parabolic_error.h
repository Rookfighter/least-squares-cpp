
#ifndef LSQCPP_PARABOLIC_ERROR_
#define LSQCPP_PARABOLIC_ERROR_

struct ParabolicError
{
    constexpr static bool ComputesJacobian = true;

    template<typename I, typename O, typename J>
    void operator()(const Eigen::MatrixBase<I> &xval,
                    Eigen::MatrixBase<O> &fval,
                    Eigen::MatrixBase<J> &jacobian) const
    {
        assert(xval.size() % 2 == 0);

        // calculate the error vector
        fval.derived().resize(xval.size() / 2);
        for(lsqcpp::Index i = 0; i < fval.size(); ++i)
            fval(i) = xval(i*2) * xval(i*2) + xval(i*2+1) * xval(i*2+1);

        // calculate the jacobian explicitly
        jacobian.derived().setZero(fval.size(), xval.size());
        for(lsqcpp::Index i = 0; i < jacobian.rows(); ++i)
        {
            jacobian(i, i*2) = 2* xval(i*2);
            jacobian(i, i*2+1) = 2* xval(i*2+1);
        }
    }
};

struct ParabolicErrorNoJacobian
{
    constexpr static bool ComputesJacobian = false;

    template<typename I, typename O>
    void operator()(const Eigen::MatrixBase<I> &xval,
                    Eigen::MatrixBase<O> &fval) const
    {
        assert(xval.size() % 2 == 0);

        // calculate the error vector
        fval.derived().resize(xval.size() / 2);
        for(lsqcpp::Index i = 0; i < fval.size(); ++i)
            fval(i) = xval(i*2) * xval(i*2) + xval(i*2+1) * xval(i*2+1);
    }
};

struct ParabolicErrorInverseJacobian
{
    constexpr static bool ComputesJacobian = true;

    template<typename I, typename O, typename J>
    void operator()(const Eigen::MatrixBase<I> &xval,
                    Eigen::MatrixBase<O> &fval,
                    Eigen::MatrixBase<J> &jacobian) const
    {
        ParabolicError error;
        error(xval, fval, jacobian);
        jacobian *= -1;
    }
};

#endif