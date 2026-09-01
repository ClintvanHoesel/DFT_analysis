import itertools
import os
from collections.abc import Iterable
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import interp1d


def get_function(name: str, module: Mapping[str, Any] | Any = globals()):
    """Resolve a function name from a mapping or module-like object."""
    if isinstance(module, Mapping):
        try:
            return module[name]
        except KeyError as exc:
            raise AttributeError(f"Function {name!r} was not found") from exc
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"Function {name!r} was not found") from exc


def sort_x(x, *y):
    """Process sort x."""
    mask = np.argsort(x)
    x = x[mask]
    yout = [yi[mask] for yi in y]
    return x, *yout


def f_extrapolate(x, y, fill_value="extrapolate", extrap_axis=0, **kwargs):
    """Process f extrapolate."""
    f = interp1d(x, y, fill_value=fill_value, axis=extrap_axis, **kwargs)
    return f


def extrapolate(xnew, x, y, **kwargs):
    """Process extrapolate."""
    return f_extrapolate(x, y, **kwargs)(xnew)


def extrapolate_concatenate(xnew, x, y, new_sort=True, conc_axis=0, **kwargs):
    """Process extrapolate concatenate."""
    ynew = extrapolate(xnew, x, y, **kwargs)
    xout = np.concatenate((xnew, x), axis=conc_axis)
    yout = np.concatenate((ynew, y), axis=conc_axis)
    if new_sort:
        xout, yout = sort_x(xout, yout)
    return xout, yout


def extrapolate_diff(x, diff_sort=False, new_sort=False, **kwargs):
    """Process extrapolate diff."""
    if diff_sort:
        mask = np.argsort(x)
        x = x[mask]
    xnew = np.expand_dims(x[0], 0)
    x, dx = extrapolate_concatenate(
        xnew, x[1:], np.diff(x), new_sort=new_sort, **kwargs
    )
    return x, dx


def get_diffed_y(x, y, new_sort=False, **kwargs):
    """Return diffed y."""
    x, dx = extrapolate_diff(x, new_sort=new_sort, **kwargs)
    y, dy = extrapolate_diff(y, new_sort=new_sort, **kwargs)

    return dx, dy


def ensure_folder(path):
    """Process ensure folder."""
    if not os.path.exists(path):
        os.makedirs(path)


def concatenate_data(*x, fv=np.nan):
    """Process concatenate data."""
    return list(itertools.zip_longest(*x, fillvalue=fv))


def round_to_nearest_significant_digit(x):
    """Process round to nearest significant digit."""
    if x == 0:
        return 0
    pow10 = np.floor(np.log10(x))
    xtemp = x * (10 ** (-pow10))
    return np.ceil(xtemp) * (10 ** (pow10))


def zip_dicts(*dcts):
    """Process zip dicts."""
    if not dcts:
        return
    for i in set(dcts[0]).intersection(*dcts[1:]):
        out = (i,) + tuple(d[i] for d in dcts)
        yield out


def flatten(xs):
    """Process flatten."""
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def check_true(s):
    """Process check true."""
    if "t" in s.lower():
        return True
    else:
        return False


def approx_equals(v1, v2, eps=1e-10):
    """Process approx equals."""
    scale = np.maximum(np.maximum(np.abs(v1), np.abs(v2)), np.finfo(float).eps)
    return np.abs(v1 - v2) / scale <= eps


def check_false(s):
    """Process check false."""
    if "f" in s.lower():
        return True
    else:
        return False


def bool_string(*s):
    """Process bool string."""
    out = []
    for val in s:
        if check_false(val):
            if check_true(val):
                raise ValueError("Both True and False found.")

            else:
                out.append(False)
        elif check_true(val):
            out.append(True)
        else:
            raise ValueError("Not a boolean value.")
    if len(out) == 1:
        out = out[0]
    elif len(out) == 0:
        raise ValueError("Could not find boolean.")
    return out


def transpose_list_of_lists(l, fv=None, **kwargs):
    """Process transpose list of lists."""
    return concatenate_data(*l, fv=fv, **kwargs)


def np_to_tuple(a):
    """Process np to tuple."""
    try:
        return tuple(np_to_tuple(i) for i in a)
    except TypeError:
        return a


def unique_mask(x, *args):
    """Process unique mask."""
    __, ind = np.unique(x, return_index=True)
    return ind


def sort_mask(x, *args):
    """Process sort mask."""
    ind = np.argsort(x)
    return ind


def n_uniques(x):
    """Process n uniques."""
    return len(set(x))


def apply_mask(x, mask, *y):
    """Process apply mask."""
    for yi in y:
        assert yi.shape == x.shape
    x = x[mask]
    yout = (yi[mask] for yi in y)
    return x, *yout


def mask_apply(x, *y, f=sort_mask):
    """Process mask apply."""
    mask = f(x)
    x, *y = apply_mask(x, mask, *y)
    return x, *y


def dic_to_arr(d, shape=None, f=np.zeros):
    """Process dic to arr."""
    k = np.array(tuple(d.keys()))
    if k.ndim == 1:

        k = k[:, None]

    v = np.array(list(d.values()))
    if shape is None:
        k2 = np.array(k)
        assert k2.ndim == 2
        shape = np.max(k2, axis=0) + 1
    arr = f(shape, dtype=v.dtype)

    arr[tuple(k.T)] = v
    return arr


def load_csv_Clint(f):
    """Process load csv Clint."""
    data = np.loadtxt(f, delimiter=",")
    data = np.array(mask_apply(*data.T)).T
    data = np.array(mask_apply(*data.T, f=unique_mask)).T
    return data


def concatenate_with_zeros(A, B, f=np.zeros):
    """Process concatenate with zeros."""
    i, j = A.shape
    k, l = B.shape

    result = np.zeros((i + k, j + l))

    result[:i, :j] = A

    result[i:, j:] = B

    return result


def concatenate_with_zeros_block(A, B, f=np.zeros):
    """Process concatenate with zeros block."""
    result = np.block(
        [
            [
                A,
                f((A.shape[0], B.shape[1])),
            ],
            [
                f((B.shape[0], A.shape[1])),
                B,
            ],
        ]
    )
    return result


def concatenate_with_reorgsshape(A, B, f=np.zeros):
    """Process concatenate with reorgsshape."""
    result = concatenate_with_zeros_block(A, B, f)
    result = result.reshape((A.shape[0], B.shape[0], -1))
    return result
