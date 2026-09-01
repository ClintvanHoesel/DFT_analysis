from copy import deepcopy
from datetime import datetime, timedelta

import loky
import numpy as np


class _MinGoal:
    def __init__(
        self,
        dt: timedelta | datetime | int | float = np.inf,
        npoints: int = np.inf,
        loss: float = np.inf,
    ):
        """Initialize the object."""
        self.dt = dt if isinstance(dt, (timedelta, datetime)) else timedelta(seconds=dt)
        self.start_time = None
        self.npoints = npoints
        self.loss = loss

    def __call__(self, learner):
        """Evaluate the object for the supplied arguments."""
        tg = self._timegoal(learner)
        npg = self._points_goal(learner)
        lg = self._loss_goal(learner)
        return tg or npg or lg

    def _points_goal(self, learner):
        """Handle points goal internally."""
        return learner.npoints >= self.npoints

    def _loss_goal(self, learner):
        """Handle loss goal internally."""
        return learner.loss() <= self.loss

    def _timegoal(self, _):
        """Handle timegoal internally."""
        if isinstance(self.dt, timedelta):
            if self.start_time is None:
                self.start_time = datetime.now()
            return datetime.now() - self.start_time > self.dt
        if isinstance(self.dt, datetime):
            return datetime.now() > self.dt
        raise TypeError(f"`dt={self.dt}` is not a datetime, timedelta, or number.")


def partial_deepcopy(original_dict, exclude_keys=None):
    """Process partial deepcopy."""
    if exclude_keys is None:
        exclude_keys = []

    temp_dict = {k: v for k, v in original_dict.items() if k not in exclude_keys}

    result = deepcopy(temp_dict)

    for key in exclude_keys:
        if key in original_dict:
            result[key] = original_dict[key]

    return result


def get_default_runner_kwargs(
    max_workers=16, dt=timedelta(minutes=5), npoints=int(1e5), loss=1e-3
):
    """Return default runner kwargs."""
    runner_kwargs = {
        "goal": _MinGoal(dt=dt, npoints=npoints, loss=loss),
        "executor": loky.get_reusable_executor(max_workers=max_workers),
    }
    return runner_kwargs
