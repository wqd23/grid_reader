import numpy as np
import pytest

from reader import readA


@pytest.fixture
def Afile():
    return "tests/data/211111165838_bnu01_40p0_30s_ch2_COM7-Data.txt"


def get_shape(d: dict):
    t = {}
    for k, v in d.items():
        assert isinstance(v, np.ndarray)
        t[k] = v.shape
    return t


def assert_tel_shape(d: dict):
    s = get_shape(d)
    assert len(set(s.values())) == 1


def assert_sci_shape(d: dict):
    s = get_shape(d)
    NUM = 44
    KEYS = ["channel", "adc_value", "usc"]
    for k, v in s.items():
        s[k] = v[0] * NUM if k not in KEYS else v[0]
    assert len(set(s.values())) == 1


def test_07A(Afile):
    sci, tel = readA(Afile, rtype="raw", nocache=True)
    sci, tel = readA(Afile, rtype="raw")
    assert_tel_shape(tel)
    assert tel["utc"].shape[0] == 28
    assert_sci_shape(sci)
    assert sci["utc"].shape[0] == 488
def test_plot(Afile):
    sci, tel = readA(Afile, rtype="raw")
    from reader.GRID_A_QuickProcess import plot_spec
    plot_spec(sci, "name", "show")