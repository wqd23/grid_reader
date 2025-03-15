from typing import TypedDict

import numpy as np


class SciGrid(TypedDict):
    amp: np.ndarray
    channel: np.ndarray
    utc: np.ndarray


class TelGrid(TypedDict):
    temp: np.ndarray  # C
    bias: np.ndarray  # V
    current: np.ndarray  # uA
    utc: np.ndarray


Tel4Ch = tuple[TelGrid, TelGrid, TelGrid, TelGrid]
ArrayDict = dict[str, np.ndarray]
