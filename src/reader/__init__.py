from .GRID_A_QuickProcess import read_raw
import numpy as np
from .parse_grid_data import parse_grid_data_new

from typing import TypedDict


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
# -------------------------GRID: A type-------------------------#


def correct_A_sci(sci: ArrayDict) -> SciGrid:
    data = {}
    data["amp"] = sci["adc_value"]
    data["channel"] = sci["channel"]
    data["utc"] = sci["utc"]
    return data


def correct_A_tel(tel: ArrayDict) -> Tel4Ch:
    data = ({}, {}, {}, {})
    for i, d in enumerate(data):
        d["temp"] = tel[f"sipm_temp_C{i}"]
        d["bias"] = tel[f"bias{i}"]
        d["current"] = tel[f"imon_uA{i}"]
        d["utc"] = tel["utc"]
    return data


def readA(filename: str) -> tuple[tuple[SciGrid, Tel4Ch], tuple[ArrayDict, ArrayDict]]:
    raw_sci, raw_tel = read_raw(filename)
    sci = correct_A_sci(raw_sci)
    tel = correct_A_tel(raw_tel)
    return (sci, tel), (raw_sci, raw_tel)


# -------------------------GRID: B type-------------------------#
B_XML = {"11B": "src/reader/frame/grid_packet_11B.xml"}
DATA_TAGS = {
    "feat": "grid1x_ft_packet",
    "wave": "grid1x_wf_packet",
    "hk": "hk_grid1x_packet",
}
ENDIAN = "MSB"


def correct_voltage(vol: np.ndarray):
    # convert to V
    return vol / 1000


def correct_sipm_temp(temp: np.ndarray):
    # convert to C
    return temp / 100 - 273.15


def correct_current(cur: np.ndarray):
    # convert to uA
    return cur


def correct_11B_tel(tel: ArrayDict) -> Tel4Ch:
    data = ({}, {}, {}, {})
    for i, d in enumerate(data):
        d["temp"] = correct_sipm_temp(tel[f"sipm_temp{i}"])
        d["bias"] = correct_voltage(tel[f"sipm_voltage{i}"])
        d["current"] = correct_current(tel[f"sipm_current{i}"])
    return data


def correct_11B_sci(sci: ArrayDict) -> SciGrid:
    data = {}
    data["amp"] = sci["data_max"] - sci["data_base"] / 4
    data["channel"] = sci["channel_n"]
    data["utc"] = sci["utc"]
    return data


def read11B(filename: str, dtype="feat") -> tuple[SciGrid | Tel4Ch, ArrayDict]:
    """
    Read GRID B type data from a file.

    Parameters
    ----------
    filename : str
        The file name of the data file.
    dtype : str, optional
        The type of the data. Should be in {"feat", "wave", "hk"}.
        Default is "feat".

    Returns
    -------
    data : SciGrid | Tel4Ch
        processed data
    raw_data : ArrayDict
        raw data from the file
    """
    VER = "11B"
    tag = DATA_TAGS.get(dtype, None)
    assert tag is not None, (
        f"unknown type: {dtype}, should be in {list(DATA_TAGS.keys())}"
    )
    xml = B_XML.get(VER, None)
    assert xml is not None, (
        f"unknown version: {filename}, should be in {list(B_XML.keys())}"
    )
    raw_data = parse_grid_data_new(filename, xml_file=xml, data_tag=tag, endian=ENDIAN)[
        0
    ]
    match dtype:
        case "hk":
            data = correct_11B_tel(raw_data)
        case _:
            data = correct_11B_sci(raw_data)
    return data, raw_data
