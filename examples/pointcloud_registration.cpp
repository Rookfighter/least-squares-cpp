/// solve_icp.h
///
/// Author:     Fabian Meyer
/// Created On: 22 Jul 2019
/// License:    MIT

#include <lsqcpp/lsqcpp.h>
#include <fstream>

using Pointcloud = Eigen::Matrix<float, 3, Eigen::Dynamic>;
using Vector6 = Eigen::Matrix<float, 6, 1>;
using Vector3 = Eigen::Matrix<float, 3, 1>;
using Matrix3 = Eigen::Matrix<float, 3, 3>;
using JacobiMatrix = Eigen::Matrix<float, Eigen::Dynamic, 6>;
using VectorX = Eigen::Matrix<float, Eigen::Dynamic, 1>;

static Pointcloud loadPointcloud(const std::string &path)
{
    std::ifstream is(path);
    std::string line;
    std::vector<Eigen::Vector3f> points;

    while(std::getline(is, line))
    {
        const auto pos1 = line.find(',');
        const auto pos2 = line.find(',', pos1 + 1);

        if(pos1 == std::string::npos || pos2  == std::string::npos)
            continue;

        char *endptr = nullptr;
        const auto x = static_cast<float>(std::strtod(line.substr(0, pos1).c_str(), &endptr));
        if(endptr == nullptr)
            continue;
        const auto y = static_cast<float>(std::strtod(line.substr(pos1 + 1, pos2).c_str(), &endptr));
        if(endptr == nullptr)
            continue;
        const auto z = static_cast<float>(std::strtod(line.substr(pos2 + 1).c_str(), &endptr));
        if(endptr == nullptr)
            continue;
        points.push_back({x, y, z});
    }

    Pointcloud result(3, points.size());
    for(lsqcpp::Index i = 0; i < result.cols(); ++i)
    {
        result.col(i) = points[i];
    }

    return result;
}

static void savePointcloud(const std::string &path, const Pointcloud &pointcloud)
{
    std::ofstream os(path);

    for(lsqcpp::Index i = 0; i < pointcloud.cols(); ++i)
    {
        os << pointcloud(0, i) << ',' <<  pointcloud(1, i) << ','  <<  pointcloud(2, i) << std::endl;
    }
}


struct Callback
{
    Callback(Pointcloud &pointcloudA, Pointcloud &pointcloudB)
        : pointcloudA(&pointcloudA), pointcloudB(&pointcloudB)
    { }

    Pointcloud *pointcloudA = nullptr;
    Pointcloud *pointcloudB = nullptr;

    bool operator()(const lsqcpp::Index iteration,
                    const Vector6& xval,
                    const VectorX&,
                    const JacobiMatrix&,
                    const Vector6&,
                    const Vector6&)
    {
        Vector3 trans = xval.segment(0, 3);
        Matrix3 rot = lsqcpp::parameter::decodeRotation(xval.segment(3, 3));

        Pointcloud cloud(pointcloudB->rows(), pointcloudB->cols());
        for(lsqcpp::Index i = 0; i < pointcloudB->cols(); ++i)
        {
            cloud.col(i) = rot * pointcloudB->col(i) + trans;
        }

        std::stringstream ss;
        ss << std::setw(3) << std::setfill('0') << iteration << "_pointcloud.b.csv";
        savePointcloud(ss.str(), cloud);

        ss.str("");
        ss << std::setw(3) << std::setfill('0') << iteration << "_pointcloud.a.csv";
        savePointcloud(ss.str(), *pointcloudA);

        return true;
    }
};

struct Objective
{
    constexpr static bool ComputesJacobian = false;

    Objective() = default;

    Objective(Pointcloud &pointcloudA, Pointcloud &pointcloudB)
        : pointcloudA(&pointcloudA), pointcloudB(&pointcloudB)
    { }

    Pointcloud *pointcloudA = nullptr;
    Pointcloud *pointcloudB = nullptr;

    template<typename Scalar, int Inputs, int Outputs>
    void operator()(const Eigen::Matrix<Scalar, Inputs, 1> &xval,
                    Eigen::Matrix<Scalar, Outputs, 1> &fval) const
    {
        Vector3 translation = xval.segment(0, 3);
        Matrix3 rotation = lsqcpp::parameter::decodeRotation(xval.segment(3, 3));

        fval.resize(pointcloudA->cols());
        for(lsqcpp::Index i = 0; i < pointcloudA->cols(); ++i)
        {
            fval(i) = (pointcloudA->col(i) - (rotation * pointcloudB->col(i) + translation)).norm();
        }
    }

};

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cerr << "usage: pointcloud_registration <file1> <file2>" << std::endl;
        return 1;
    }

    auto pointcloudA = loadPointcloud(argv[1]);
    auto pointcloudB = loadPointcloud(argv[2]);

    auto callback = Callback(pointcloudA, pointcloudB);

    lsqcpp::GaussNewton<float, 6, Eigen::Dynamic, Objective, lsqcpp::ArmijoBacktracking, lsqcpp::DenseCholeskySolver> optimizer;
    optimizer.setMinimumGradientLength(1e-3);
    optimizer.setMinimumStepLength(1e-3);
    optimizer.setObjective({pointcloudA, pointcloudB});
    optimizer.setCallback(callback);
    optimizer.setMaximumIterations(10);
    optimizer.setVerbosity(3);

    Vector6 xval(6);
    xval.setZero();
    xval.segment(3, 3) = lsqcpp::parameter::encodeRotation(Matrix3::Identity());

    optimizer.minimize(xval);

    return 0;
}