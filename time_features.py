'''
Modified version of https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_self_supervised/src/data/timefeatures.py
'''

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5
    
class Year(TimeFeature):
    """Year from 2010-2020 encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().year - 2010) / 10.0 - 0.5


class TimeFeatures():
    def __init__(self, use_features):
        self.features_by_offsets = []

        if 'SecondOfMinute' in use_features:
                self.features_by_offsets.append(SecondOfMinute)
        if 'MinuteOfHour' in use_features:
                self.features_by_offsets.append(MinuteOfHour)
        if 'HourOfDay' in use_features:
                self.features_by_offsets.append(HourOfDay)
        if 'DayOfWeek' in use_features:
                self.features_by_offsets.append(DayOfWeek)
        if 'DayOfMonth' in use_features:
                self.features_by_offsets.append(DayOfMonth)
        if 'DayOfYear' in use_features:
                self.features_by_offsets.append(DayOfYear)
        if 'WeekOfYear' in use_features:
                self.features_by_offsets.append(WeekOfYear)
        if 'MonthOfYear' in use_features:
                self.features_by_offsets.append(MonthOfYear)
        if 'Year' in use_features:
                self.features_by_offsets.append(Year)

    def time_features_from_frequency_str(self) -> List[TimeFeature]:
        """
        Returns a list of time features that will be appropriate for the given frequency string.
        Parameters
        ----------
        freq_str
            Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
        """

        return [cls() for cls in self.features_by_offsets]

    def time_features(self, dates):
        return np.stack([feat(dates) for feat in self.time_features_from_frequency_str()]).T
