#!/usr/bin/env python3

import argparse
from itertools import product

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pyDOE import lhs
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
from sklearn.preprocessing import StandardScaler


aspect = 1/1.618
resolution = 72.27
columnwidth = 246/resolution
textwidth = 510/resolution
textiny, texsmall, texnormal = 8.0, 9.25, 10.0
offblack = '#262626'

plt.rcdefaults()
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Lato'],
    'mathtext.fontset': 'custom',
    'mathtext.default': 'it',
    'mathtext.rm': 'sans',
    'mathtext.cal': 'sans',
    'font.family': 'serif',
    'font.size': texsmall,
    'legend.fontsize': texsmall,
    'axes.labelsize': texsmall,
    'axes.titlesize': texsmall,
    'xtick.labelsize': textiny,
    'ytick.labelsize': textiny,
    'font.weight': 400,
    'axes.labelweight': 400,
    'axes.titleweight': 400,
    'lines.linewidth': .9,
    'lines.markersize': 3,
    'lines.markeredgewidth': .1,
    'patch.linewidth': .9,
    'axes.linewidth': .5,
    'xtick.major.width': .5,
    'ytick.major.width': .5,
    'xtick.minor.width': .5,
    'ytick.minor.width': .5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'xtick.minor.size': 1.3,
    'ytick.minor.size': 1.3,
    'xtick.major.pad': 1.8,
    'ytick.major.pad': 1.8,
    'text.color': 'black',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'legend.frameon': False,
    'image.interpolation': 'none',
    'pdf.fonttype': 42,
})


plot_functions = {}

def plot(f):
    def wrapper(*args, **kwargs):
        print(f.__name__)
        f(*args, **kwargs)
        plt.savefig('{}.pdf'.format(f.__name__))
        plt.close()

    plot_functions[f.__name__] = wrapper

    return wrapper


def finish(despine=True, remove_ticks=False, pad=0.1, h_pad=None, w_pad=None,
           rect=[0, 0, 1, 1]):
    fig = plt.gcf()

    for ax in fig.axes:
        if despine:
            for spine in 'top', 'right':
                ax.spines[spine].set_visible(False)

        if remove_ticks:
            for ax_name in 'xaxis', 'yaxis':
                getattr(ax, ax_name).set_ticks_position('none')
        else:
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

    fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)


class Emulator:
    def __init__(self, npc=3, nobs=10**2):

        # Design
        # np.random.seed(123)
        self.npc = npc
        self.nobs = nobs
        self.bins = np.arange(nobs)
        self.X = lhs(2, samples=400, criterion="cm")
        self.Y = np.array([self.model(*x) for x in self.X])

        # Principal components
        self.scaler = StandardScaler(copy=False)
        self.pca = PCA(copy=False, whiten=True, svd_solver='full')
        self.Z = self.pca.fit_transform(self.scaler.fit_transform(self.Y))

        # Train emulator
        self.gps = [self.train_gp(self.X, z) for z in self.Z.T[:npc]]

    def model(self, alpha, beta):
        """
        Simple toy-model where,

        f(x; alpha, beta) =
            alpha*x + small error:  if x < 50
            beta + large error:     if x >= 50

        """
        x1, x2 = np.split(self.bins, 2)
        noise1, noise2 = (.1, 0.02)

        y1 = alpha*np.ones_like(x1) + beta*x2/100. + noise1*np.random.rand(len(x1))
        y2 = alpha*np.ones_like(x2) + noise2*np.random.rand(len(x2))

        return np.concatenate((y1, y2))

    def train_gp(self, X, y):
        """
        Trains a Gaussian process emulator y = F(X)

        """
        kernel = (
            1. * kernels.RBF(
                length_scale=1,
                length_scale_bounds=(.1, 10),
            )
            +
            kernels.WhiteKernel(
                noise_level=.1**2,
                noise_level_bounds=(0.01**2, 1),
            )
        )

        return GPR(
            kernel=kernel, alpha=0,
            n_restarts_optimizer=0,
            copy_X_train=False
        ).fit(X, y)

    def predict(self, x):
        """
        Predict y at location x using the emulator

        """
        Z = np.array([gp.sample_y(x, n_samples=10**3) for gp in self.gps]).T

        size = Z.shape[:-1] + (self.nobs - self.npc,)
        R = np.random.normal(size=size)

        Z = np.concatenate((Z, R), axis=2)
        Y = self.scaler.inverse_transform(self.pca.inverse_transform(Z))

        return np.squeeze(Y.mean(axis=0)), np.squeeze(Y.std(axis=0))


emu = Emulator()


@plot
def pc_design(emu=emu):
    """
    Plots the values of each PC as a function of the design parameters X

    """
    figsize = (textwidth, aspect*columnwidth)
    fig = plt.figure(figsize=plt.figaspect(1/emu.npc))

    X = emu.X
    Y = emu.Y
    Z = emu.Z

    alpha, beta = zip(*X)

    Zemu = np.array([gp.sample_y(X, n_samples=10**3) for gp in emu.gps]).T
    Z0 = Zemu.mean(axis=0)
    dZ0 = Zemu.std(axis=0)

    for n, (z, z0, dz0) in enumerate(zip(Z.T, Z0.T, dZ0.T), start=1):
        ax = fig.add_subplot(1, emu.npc, n, projection='3d')
        ax.scatter(alpha, beta, z)
        ax.scatter(alpha, beta, z0 - 2*dz0)
        ax.scatter(alpha, beta, z0 + 2*dz0)

        ax.set_xlabel('α')
        ax.set_ylabel('β')
        ax.set_zlabel('PC{}'.format(n))
        ax.view_init(15, 200)

    plt.tight_layout()


@plot
def validate_y(emu=emu):
    """
    Compare emulator's prediction to the model output at position x0

    """
    figsize = (columnwidth, aspect*columnwidth)
    fig = plt.figure(figsize=figsize)

    blue = plt.cm.Blues(.6)
    orange = plt.cm.Oranges(.6)

    x0 = [0.5, 0.2]
    y_model = emu.model(*x0)
    plt.scatter(emu.bins, y_model, color=orange, label='model')

    y_emu, dy_emu = emu.predict([x0])
    plt.scatter(emu.bins, y_emu, color=blue, label='emulator')
    plt.fill_between(
        emu.bins, y_emu - 2*dy_emu, y_emu + 2*dy_emu,
        color=blue, alpha=.2, lw=0, zorder=0
    )

    plt.xlabel('Observable')
    plt.ylabel('$y$')
    plt.legend(handletextpad=0)

    finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plots', nargs='*')
    args = parser.parse_args()

    if args.plots:
        for i in args.plots:
            if i.endswith('.pdf'):
                i = i[:-4]
            if i in plot_functions:
                plot_functions[i]()
            else:
                print('unknown plot:', i)
    else:
        for f in plot_functions.values():
            f()


if __name__ == "__main__":
    main()

