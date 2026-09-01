import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import BSpline, interp1d, splrep

from .base_utils import apply_mask, mask_apply, sort_mask, unique_mask


class Data_Series:
    def __init__(
        self,
        x=None,
        y=None,
        function_type="scipy",
        weight=1.0,
        kind="linear",
        left=0.0,
        right=0.0,
        fkwargs=None,
    ):
        """Initialize the object."""
        self._x = x
        self._y = y
        self._data_processed = False

        self.function_type = function_type

        self.kind = kind
        self.fkwargs = {} if fkwargs is None else fkwargs
        self.weight = weight
        self.left = left
        self.right = right

    def _process_data(self):
        """Process x and y data to ensure x is sorted with unique values"""
        if self._x is None or self._y is None:
            self._data_processed = True
            return

        self._x, self._y = mask_apply(self._x, self._y, f=sort_mask)
        self._x, self._y = mask_apply(self._x, self._y, f=unique_mask)
        self._data_processed = True

    @property
    def x(self):
        """Process x."""
        if not self._data_processed:
            self._process_data()
        return self._x

    @x.setter
    def x(self, value):
        """Process x."""
        if value is not None and not isinstance(value, np.ndarray):
            value = np.array(value)
        self._x = value
        self._data_processed = False

    @property
    def y(self):
        """Process y."""
        if not self._data_processed:
            self._process_data()
        return self._y

    @y.setter
    def y(self, value):
        """Process y."""
        if value is not None and not isinstance(value, np.ndarray):
            value = np.array(value)
        self._y = value
        self._data_processed = False

    @property
    def function(self):
        """Process function."""
        if self.function_type.lower() == "scipy":
            return interp1d(
                self.x,
                self.y,
                kind=self.kind,
                bounds_error=False,
                fill_value=(self.left, self.right),
                **self.fkwargs,
            )
        elif self.function_type.lower() == "bspline":
            tck = splrep(self.x, self.y, **self.fkwargs)
            return BSpline(*tck)
        else:
            return lambda xinp: np.interp(
                xinp,
                self.x,
                self.y,
                left=self.left,
                right=self.right,
                **self.fkwargs,
            )

    def get_value(self, x, *args, **kwargs):
        """Return value."""
        return self.function(x, *args, **kwargs)

    def __call__(self, x, *args, **kwargs):
        """Evaluate the object for the supplied arguments."""
        return self.get_value(x, *args, **kwargs)

    @property
    def norm(self):
        """Process norm."""
        return trapezoid(y=self.y, x=self.x)

    def __getstate__(self):
        """
        Return only the attributes needed to rebuild the series.
        NumPy arrays, floats and dicts of simple params are pickleable.
        """
        state = {
            "_x": self._x,
            "_y": self._y,
            "_data_processed": self._data_processed,
            "function_type": self.function_type,
            "kind": self.kind,
            "fkwargs": self.fkwargs,
            "weight": self.weight,
            "left": self.left,
            "right": self.right,
        }
        return state

    def __setstate__(self, state):
        """
        Restore the minimal state, re‑initializing any internal caches
        on first use (via your existing property‑based logic).
        """
        self._x = state["_x"]
        self._y = state["_y"]
        self._data_processed = state["_data_processed"]
        self.function_type = state["function_type"]
        self.kind = state["kind"]
        self.fkwargs = state["fkwargs"]
        self.weight = state["weight"]
        self.left = state["left"]
        self.right = state["right"]

    def __deepcopy__(self, memo):
        """Handle deepcopy internally."""
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        state = self.__getstate__()

        import copy

        for k, v in state.items():
            setattr(new, k, copy.deepcopy(v, memo))

        return new


def preserve_first_false(b: np.ndarray) -> np.ndarray:
    """
    Turn all False entries of b into True, except keep the first False as-is.

    Parameters
    ----------
    b : np.ndarray of bool, shape (n,)
        Input boolean array.

    Returns
    -------
    np.ndarray of bool, shape (n,)
        New array where only the first False remains False.
    """

    out = b.copy()

    false_idx = np.flatnonzero(~b)
    if false_idx.size > 1:

        out[false_idx[1:]] = True

    return out


def get_index_first_false(b: np.array) -> int:
    """
    Get the index of the first False entry in a boolean array.

    Parameters
    ----------
    b : np.ndarray of bool, shape (n,)
        Input boolean array.

    Returns
    -------
    int
        Index of the first False entry, or -1 if no False entries are found.
    """

    false_idx = np.flatnonzero(~b)
    if false_idx.size > 0:
        return false_idx[0]
    else:
        return -1


def mask_between_first_falses(b: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Turn all False entries of b into True, except keep the first False as-is.

    Parameters
    ----------
    b : np.ndarray of bool, shape (n,)
        Input boolean array.

    Returns
    -------
    np.ndarray of bool, shape (n,)
        New array where only the first False remains False.
    """

    out = np.ones_like(b, dtype=bool)

    false_idx1 = get_index_first_false(b)
    false_idx2 = get_index_first_false(b2)
    if false_idx1 == -1 or false_idx2 == -1:
        return out
    else:
        out[min(false_idx1, false_idx2) : max(false_idx1, false_idx2)] = False

    return out


def _find_spike_mask(y, x=None, threshold=3.0, min_val=None, window=11):
    """Find spikes in the data"""
    mask = np.ones_like(y, dtype=bool)
    n = y.size
    if n < 2:
        return mask

    d = np.abs(np.diff(y))
    half_window = window // 2
    min_val = np.finfo(np.float64).eps if min_val is None else min_val

    dp = np.pad(d, (half_window, half_window), mode="edge")
    wins = np.lib.stride_tricks.sliding_window_view(dp, window)
    med_d = np.median(wins, axis=1)

    threshold = threshold * med_d
    spikes = np.nonzero(d > threshold)[0] + 1
    mask[spikes] = False

    return mask


def remove_spikes(y, x=None, threshold=5.0, min_val=None):
    """Remove spikes from the data"""
    mask = _find_spike_mask(y, x=x, threshold=threshold, min_val=min_val)

    i = 0
    while not np.all(mask):
        i += 1
        mask = preserve_first_false(mask)
        x, y = apply_mask(x, mask, y)
        mask = _find_spike_mask(y, x, threshold=threshold, min_val=min_val)

    return x, y


class Data_Series_removeSpikes(Data_Series):
    def __init__(
        self,
        x=None,
        y=None,
        function_type="scipy",
        weight=1.0,
        kind="linear",
        left=0.0,
        right=0.0,
        fkwargs=None,
        threshold=5.0,
        min_val=None,
    ):
        """Initialize the object."""
        self.threshold = threshold
        if min_val is None:
            nonzero = np.abs(y)[np.abs(y) > 0]
            min_val = np.min(nonzero) if nonzero.size else np.finfo(float).eps
        self.min_val = min_val
        super().__init__(
            x=x,
            y=y,
            function_type=function_type,
            weight=weight,
            kind=kind,
            left=left,
            right=right,
            fkwargs=fkwargs,
        )

    def _process_data(self):
        """Process x and y data to ensure x is sorted with unique values"""
        if self._x is None or self._y is None:
            self._data_processed = True
            return

        self._x, self._y = mask_apply(self._x, self._y, f=sort_mask)
        self._x, self._y = mask_apply(self._x, self._y, f=unique_mask)

        self._x, self._y = remove_spikes(
            self._y,
            x=self._x,
            threshold=self.threshold,
            min_val=self.min_val,
        )

        self._data_processed = True


class Averager_Data_Series:
    def __init__(
        self,
        data_series=None,
        trial_point=0.0,
        mask_array=True,
        fill_zero_n=np.nan,
    ):
        """Initialize the object."""
        self.trial_point = trial_point
        self.data_series = [] if data_series is None else list(data_series)
        [f(self.trial_point) for f in self.data_series]
        self.mask_array = mask_array
        self.fill_zero_n = fill_zero_n

    def add(self, data_serie):
        """Process add."""
        data_serie(self.trial_point)
        self.data_series.append(data_serie)

    def __iadd__(self, x):
        """Handle iadd internally."""
        self.add(x)
        return self

    @property
    def n(self):
        """Process n."""
        return len(self.data_series)

    @property
    def inv_sqrtn(self):
        """Process inv sqrtn."""
        return self.n ** (-0.5)

    @property
    def weights(self):
        """Process weights."""
        return [f.weight for f in self.data_series]

    def get_values(self, x, *args, **kwargs):
        """Return values."""
        return [f(x, *args, **kwargs) for f in self.data_series]

    def get_masked_values(self, x, *args, **kwargs):
        """Return masked values."""
        vals = self.get_values(x, *args, **kwargs)

        if self.mask_array:
            vals = np.ma.masked_array(vals, np.isnan(vals))
        return vals

    def get_mean(self, x, *args, **kwargs):
        """Return mean."""
        vals = self.get_masked_values(x, *args, **kwargs)
        return np.average(vals, axis=0)

    def get_weighted_mean(self, x, *args, **kwargs):
        """Return weighted mean."""
        vals = self.get_masked_values(x, *args, **kwargs)
        return np.average(vals, weights=self.weights, axis=0)

    def get_std(self, x, *args, **kwargs):
        """Return std."""
        vals = self.get_masked_values(x, *args, **kwargs)
        return np.std(vals, axis=0)

    def get_weighted_std(self, x, *args, **kwargs):
        """Return weighted std."""
        vals = self.get_masked_values(x, *args, **kwargs)
        av = np.average(vals, weights=self.weights, axis=0)
        return np.sqrt(
            np.average(
                (vals - np.expand_dims(av, axis=0)) ** 2, weights=self.weights, axis=0
            )
        )

    def n_x(self, x, *args, **kwargs):
        """Process n x."""
        vals = self.get_masked_values(x, *args, **kwargs)
        vals = np.sum(np.logical_not(vals.mask), axis=0)
        return vals

    def inv_sqrtn_x(self, x, *args, **kwargs):
        """Process inv sqrtn x."""
        return 1.0 / np.sqrt(self.n_x(x, *args, **kwargs))

    def get_std_mean(self, x, *args, **kwargs):
        """Return std mean."""
        return self.get_std(x, *args, **kwargs) * self.inv_sqrtn(x, *args, **kwargs)

    def get_weighted_std_mean(self, x, *args, **kwargs):
        """Return weighted std mean."""
        return self.get_weighted_std(x, *args, **kwargs) * self.inv_sqrtn(
            x, *args, **kwargs
        )

    def get_all(self, x, *args, **kwargs):
        """Return all."""
        vals = self.get_masked_values(x, *args, **kwargs)

        n_x = np.sum(np.logical_not(vals.mask), axis=0)
        inv_n_x = 1.0 / np.sqrt(n_x)
        mean = np.average(vals, axis=0)
        std = np.std(vals, axis=0)
        wmean = np.average(vals, weights=self.weights, axis=0)
        wstd = np.sqrt(
            np.average(
                (vals - np.expand_dims(wmean, axis=0)) ** 2,
                weights=self.weights,
                axis=0,
            )
        )
        nan_mask = n_x == 0
        out = np.array(
            [
                mean,
                std,
                std * inv_n_x,
                mean - 2 * std * inv_n_x,
                mean + 2 * std * inv_n_x,
                wmean,
                wstd,
                wstd * inv_n_x,
                wmean - 2 * wstd * inv_n_x,
                wmean + 2 * wstd * inv_n_x,
                n_x,
            ]
        )
        out[:, nan_mask] = self.fill_zero_n
        return out

    def __call__(self, x, *args, **kwargs):
        """Evaluate the object for the supplied arguments."""
        return self.get_all(x, *args, **kwargs)
