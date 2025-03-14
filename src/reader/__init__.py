from .GRID_A_QuickProcess import read_raw
import numpy as np


def readA(filename: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return read_raw(filename)
