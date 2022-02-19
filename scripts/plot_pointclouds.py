import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def parse_args():
    parser = argparse.ArgumentParser(description='Plot point clouds')
    parser.add_argument('files', type=str, nargs='+', help='an integer for the accumulator')
    parser.add_argument('--nogui', action='store_true', help='disables GUI display)')

    return parser.parse_args()

def process_pointcloud(path_a, path_b, no_gui):
    pointcloud_a = np.loadtxt(path_a, delimiter=',')
    pointcloud_b = np.loadtxt(path_b, delimiter=',')

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter(pointcloud_a[:, 0],   # x
                pointcloud_a[:, 1],   # y
                pointcloud_a[:, 2],   # z
                marker=".")

    ax.scatter(pointcloud_b[:, 0],   # x
                pointcloud_b[:, 1],   # y
                pointcloud_b[:, 2],   # z
                marker=".")

    ax.set_xlim(-2, 17)
    ax.set_ylim(-2.5, 5)
    ax.set_zlim(0, 15)
    ax.view_init(54, -17)

    path = f'{path_a}_{path_b}.png'
    print(path)
    plt.savefig(path)
    if not no_gui:
        plt.show()
    plt.close()

def main():
    args = parse_args()

    if len(args.files) % 2 != 0:
        print("must have pairs of files")
        sys.exit(1)

    file_pairs = list(zip(args.files[0:-1:2], args.files[1::2]))

    for path_a, path_b in file_pairs:
        process_pointcloud(path_a, path_b, args.nogui)


if __name__ == '__main__':
    main()