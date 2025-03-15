import numpy as np

from .GRID_A_QuickProcess import read_raw
from .parse_grid_data import parse_grid_data_new
from .utils import disk_cache


from .types import SciGrid, Tel4Ch, ArrayDict

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


@disk_cache(cache_dir=".cache")
def readA(
    filename: str, rtype="sci"
) -> tuple[SciGrid, Tel4Ch] | tuple[ArrayDict, ArrayDict]:
    """
    Read GRID A type data from a file.

    Parameters
    ----------
    filename : str
        The file name of the data file.
    rtype : str, optional
        The type of the returned data. Should be in {"sci", "raw"}.
        "sci" returns the processed data in SciGrid and Tel4Ch.
        "raw" returns the raw data in ArrayDict.
        Default is "sci".

    Returns
    -------
    data : tuple[SciGrid, Tel4Ch]
        processed data
    OR
    raw_data : tuple[ArrayDict, ArrayDict]
        raw data from the file
    """

    raw_sci, raw_tel = read_raw(filename)
    sci = correct_A_sci(raw_sci)
    tel = correct_A_tel(raw_tel)
    match rtype:
        case "sci":
            return sci, tel
        case "raw":
            return raw_sci, raw_tel
        case _:
            raise ValueError(f"unknown type: {rtype}, should be in {'sci', 'raw'}")


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


@disk_cache(cache_dir=".cache")
def read11B(filename: str, dtype="feat", rtype="sci") -> SciGrid | Tel4Ch | ArrayDict:
    """
    Read GRID B type data from a file.

    Parameters
    ----------
    filename : str
        The file name of the data file.
    dtype : str, optional
        The type of the data. Should be in {"feat", "wave", "hk"}.
        Default is "feat".
    rtype : str, optional
        The type of the returned data. Should be in {"sci", "raw"}.
    Returns
    -------
    data : SciGrid | Tel4Ch
        processed data
    OR
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
    raw_data = parse_grid_data_new(filename, xml_file=xml, data_tag=tag, endian=ENDIAN)
    raw_data = raw_data[0]

    match dtype:
        case "hk":
            data = correct_11B_tel(raw_data)
        case _:
            data = correct_11B_sci(raw_data)

    match rtype:
        case "sci":
            return data
        case "raw":
            return raw_data
        case _:
            raise ValueError(f"unknown type: {rtype}, should be in {'sci', 'raw'}")
