# encoding=UTF-8
"""
    @author: Administrator on 2017/4/19
    @email: ppsunrise99@gmail.com
    @step:
    @function: 
"""
from __future__ import division
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def scatter3d(x, y, z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    ax.set_xlabel('k1')
    ax.set_ylabel('k2')
    ax.set_zlabel('k3')
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap, label='Test')
    plt.show()


def surface3d(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #    X = np.arange(11)/50
    #    Y = np.arange(11)/50
    X, Y = np.meshgrid(X, Y)
    #    Z = []
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('k2')
    ax.set_ylabel('k3')
    ax.set_zlabel('rho')
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def surface3df(df):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(11) / 50
    Y = np.arange(11) / 50
    X, Y = np.meshgrid(X[2:], Y[2:])
    Z = np.zeros(np.shape(X))
    for i in range(9):
        for j in range(9):
            line1 = df[df.k2 == X[i, j]]
            line2 = line1[line1.k3 == Y[i, j]]
            #            print '*'*5, i, j, line2.rho
            Z[i, j] = line2.rho

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # surf = plt.contour(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # plt.xlim([0.0,0.1])
    # plt.ylim([0.0,0.1])
    ax.set_zlim(0.5, 0.56)
    ax.set_xlabel('k2')
    ax.set_ylabel('k3')
    ax.set_zlabel('rho')
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def splitk1(df):
    groups = df.groupby('k1')  # 按照k1分组

    for g in groups:
        surface3df(g[1])


def combinek1(df):
    groups = df.groupby('k1')  # 按照k1分组
    print groups.sum()


def mapXYZ(df):
    n = 10
    X = np.arange(n+1) / n
    Y = np.arange(n+1) / n
    X, Y = np.meshgrid(X[:], Y[:])
    Z = np.zeros(np.shape(X))
    # k2, k3
    for i in range(n):
        for j in range(n):
            try:
                line1 = df[df.k2 == X[i, j]]
                # print 'line1', line1
                line2 = line1[line1.k3 == Y[i, j]]
                Z[i, j] = line2.rho[j]
                print len(line2.rho)
                break
                # print 'line2.rho', line2.rho
                # print '***', i, j, X[i, j], Y[i, j], Z[i, j]
            except:
                Z[i, j] = 0
                print '!!!', i, j, X[i, j], Y[i, j]
    return X, Y, Z


def mapXYZ2(df):
    X = np.arange(11) / 50
    Y = np.arange(11) / 50
    X, Y = np.meshgrid(X[:], Y[:])
    Z = np.zeros(np.shape(X))
    for i in range(11):
        for j in range(11):
            line1 = df[df.k2 == X[i, j]]
            line2 = line1[line1.k3 == Y[i, j]]
            #            print '*'*5, i, j, line2.rho
            Z[i, j] = line2.rho

    return X, Y, Z


def plot99(df):
    groups = df.groupby('k1')  # 按照k1分组
    fig = plt.figure()

    for i, g in enumerate(groups):
        print g[0], '-' * 20
        X, Y, Z = mapXYZ(g[1])
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(4, 3, 1 + i, projection='3d')
        # surf = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1) # 网格，无色差平面
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel('k2', fontsize=8)
        ax.set_ylabel('k3', fontsize=8)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(surf, cax=cbar_ax)
    plt.show()


def plot2d(df):
    groups = df.groupby('k1')  # 按照k1分组

    fig = plt.figure(figsize=(8, 8))

    for i, g in enumerate(groups):
        print g[0], '-' * 20
        X, Y, Z = mapXYZ(g[1])
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(4, 3, 1 + i)
        im = ax.imshow(Z, interpolation='nearest', origin='lower', cmap=cm.coolwarm)
        ax.set_xticks(np.linspace(0, 10, 3))
        ax.set_xticklabels([0, 0.5, 1.0], fontsize=8)
        ax.set_yticks(np.linspace(0, 10, 3))
        ax.set_yticklabels([0, 0.5, 1.0], fontsize=8)
        ax.set_title('(%s) k1=%s' % (i+1, g[0]), fontsize=11)
        ax.set_xlabel('k2', fontsize=8)
        ax.set_ylabel('k3', fontsize=8, rotation=0)
        # ax. el_coords(-0.15, 1.05)


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # plt.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    plt.show()
    # plt.savefig("fig/figure.pdf")


def plot4D(df):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = df.k1
    y = df.k2
    z = df.k3
    c = df.rho

    im = ax.scatter(x, y, z, c=c, cmap=cm.jet)
    ax.set_xlabel('k1', fontsize=8)
    ax.set_ylabel('k2', fontsize=8)
    ax.set_zlabel('k3', fontsize=8)

    fig.colorbar(im)
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(r'D:\MyCode\TALLIP_WS\data\results\hypers\hypers.txt', names=['k1', 'k2', 'k3', 'rho'],
                     delimiter='\t',
                     header=None)
    plot2d(df)
    # plot99(df)
    # plot4D(df)
    # scatter3d(df.k1, df.k2, df.k3, df.rho, colorsMap='jet')
    # surface3d(g[1].k2,g[1].k3,g[1].rho)
#    scatter3d(df[0], df[1], df[2], df[3])
