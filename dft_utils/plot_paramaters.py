"""
Created on Wed Mar  9 15:45:52 2022

@author: s164097
"""

import datetime
import os
import warnings
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm

from .base_utils import ensure_folder
from .units import Units


def set_plot_parameters(
    update_dict=None,
    nts=15.0,
    bts=18.0,
    form="jpeg",
    w=12.0,
    h=9.0,
    mats=10.0,
    mits=7.0,
    axlw=2.0,
    matlw=1.5,
    mitlw=1.5,
):
    """Set plot parameters."""
    plt.rcParams.update(
        {
            "ps.usedistiller": "xpdf",
            "text.latex.preamble": " ".join(
                [
                    r"\usepackage{amsmath}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{stix2}",
                    r"\newcommand*\mean[1]{\overline{#1}}",
                    r"\newcommand{\matr}[1]{\matrixsym{#1}}",
                    r"\newcommand{\vect}[1]{\vectorsym{#1}}",
                    r"\newcommand{\tens}[1]{\tensorsym{#1}}",
                    r"\newcommand{\of}[1]{\left(#1\right)}",
                    r"\newcommand{\off}[1]{\left[#1\right]}",
                    r"\newcommand{\offf}[1]{\left\{#1\right\}}",
                    r"\newcommand{\abss}[1]{\left|#1\right|}",
                    r"\newcommand{\innerprod}[1]{\left(#1\right)}",
                    r"\newcommand{\expec}[1]{\left<#1\right>}",
                    r"\newcommand{\tm}[1]{\text{#1}}",
                    r"\newcommand{\ofl}[1]{\left(#1\right.}",
                    r"\newcommand{\ofr}[1]{\left.#1\right)}",
                    r"\newcommand{\ofi}[1]{\left.#1\right.}",
                    r"\newcommand{\matt}[1]{\text{#1}}",
                    r"\newcommand{\funct}[1]{\matt{#1}}",
                    r"\newcommand{\diff}{\mathop{}\!\mathrm{d}}",
                    r"\newcommand{\Diff}[1]{\mathop{}\!\mathrm{d^#1}}",
                ]
            ),
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.size": nts,
            "axes.linewidth": axlw,
            "xtick.top": True,
            "xtick.bottom": True,
            "ytick.left": True,
            "ytick.right": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.top": True,
            "axes.spines.bottom": True,
            "xtick.major.size": mats,
            "ytick.major.size": mats,
            "xtick.major.width": matlw,
            "ytick.major.width": matlw,
            "xtick.minor.size": mits,
            "ytick.minor.size": mits,
            "xtick.minor.width": mitlw,
            "ytick.minor.width": mitlw,
            "xtick.labelsize": nts,
            "ytick.labelsize": nts,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.format": form,
            "savefig.transparent": True,
            "axes.titlesize": nts,
            "axes.labelsize": nts,
            "legend.fontsize": nts,
            "legend.frameon": True,
            "figure.titlesize": bts,
            "figure.labelsize": nts,
            "figure.figsize": (
                Units.convert(w, "cm", "inch"),
                Units.convert(h, "cm", "inch"),
            ),
            "figure.constrained_layout.use": True,
            "figure.dpi": 800,
        }
    )

    if update_dict:
        plt.rcParams.update(update_dict)


def geco(i, n_colours=1, cmp=cm.batlow):
    """Process geco."""
    assert n_colours > 0
    return cmp((i + 0.5) / (n_colours))


def geco2(i, n_colours=1, cmp=cm.batlow):
    """Process geco2."""
    assert n_colours >= 1
    if n_colours == 1:
        n_colours += 0.5
    return cmp((i) / (n_colours - 1))


def gecoarr(arr, interval=None, cmp=cm.batlow):
    """Process gecoarr."""
    if interval is None:
        interval = [np.min(arr), np.max(arr)]
    if interval[1] <= interval[0]:
        return cmp(np.zeros_like(arr, dtype=float))
    arr = np.clip(arr, *interval)
    arr = arr - interval[0]
    arr = arr / (interval[1] - interval[0])
    return cmp(arr)


def save_fig_many(
    fig,
    plot_name="Unknown",
    path=None,
    datstr=None,
    forms=None,
):
    """Process save fig many."""
    path = os.getcwd() if path is None else path
    datstr = (
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") if datstr is None else datstr
    )
    forms = ["png", "jpeg", "eps", "svg", "pdf", "webp"] if forms is None else forms
    if datstr:
        plot_name = plot_name + f"_{datstr}"
    ensure_folder(path)
    base_path = join(path, plot_name)
    for form in forms:
        try:
            fig.savefig(base_path + "." + form)
        except Exception as e:
            warnings.warn(
                f"Could not save figure {plot_name} with form {form}. Error: {e}",
                stacklevel=2,
            )
