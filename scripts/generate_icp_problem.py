import numpy as np
from scipy.spatial.transform import Rotation

def sample_plane_surface_xy(x, y, resolution):
    cnt = int(x * resolution * y * resolution)
    xs = np.linspace(0, x, x*resolution)
    ys = np.linspace(0, y, y*resolution)
    result = np.empty((len(xs) * len(ys), 3))

    for i in range(len(ys)):
        s = i*len(xs)
        e = (i+1)*len(xs)
        result[s:e, 0] = xs
        result[s:e, 1] = ys[i]
        result[s:e, 2] = 0
    return result


def distort_plane_points(points, normal, sigma):
    return points + normal * np.dot(sigma * np.random.randn(*points.shape), normal).reshape((points.shape[0], 1)).repeat(3, axis=1)


rot_a = Rotation.from_euler('XYZ', [90, 0, 0], degrees=True)
trans_a = np.array([0, 0, 0])

rot_ab = Rotation.from_euler('XYZ', [0, 3, 12], degrees=True)
trans_ab = np.array([0.2, 0.3, -0.1])

pointcloud_orig = sample_plane_surface_xy(15, 15, 2)
pointcloud_a_undistort = rot_a.apply(pointcloud_orig) + trans_a
pointcloud_a = distort_plane_points(pointcloud_orig, np.array([0, 0, 1]), 0.05)
pointcloud_a = rot_a.apply(pointcloud_a) + trans_a

pointcloud_b_undistort = rot_ab.apply(pointcloud_a_undistort) + trans_ab
pointcloud_b = distort_plane_points(pointcloud_orig, np.array([0, 0, 1]), 0.05)
pointcloud_b = rot_a.apply(pointcloud_b) + trans_a
pointcloud_b = rot_ab.apply(pointcloud_b) + trans_ab

np.savetxt("pointcloud.a.csv", pointcloud_a, delimiter=",")
np.savetxt("pointcloud.b.csv", pointcloud_b, delimiter=",")