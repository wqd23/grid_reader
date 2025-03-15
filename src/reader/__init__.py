from .GRID_A_QuickProcess import read_raw
import numpy as np
from .parse_grid_data import parse_grid_data_new

# -------------------------GRID: A type-------------------------#


def readA(filename: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    sci, tel = read_raw(filename)
    return sci, tel


# -------------------------GRID: B type-------------------------#
B_XML = {"11B": "src/reader/frame/grid_packet_11B.xml"}
DATA_TAGS = {
    "feat": "grid1x_ft_packet",
    "wave": "grid1x_wf_packet",
}
ENDIAN = "MSB"


def read11B(filename: str, dtype="feat"):
    VER = "11B"
    tag = DATA_TAGS.get(dtype, None)
    assert tag is not None, (
        f"unknown type: {dtype}, should be in {list(DATA_TAGS.keys())}"
    )
    xml = B_XML.get(VER, None)
    assert xml is not None, (
        f"unknown version: {filename}, should be in {list(B_XML.keys())}"
    )
    data = parse_grid_data_new(filename, xml_file=xml, data_tag=tag, endian=ENDIAN)
    return data[0]
