import pytest

from reader import read11B

from .test_A import get_shape


@pytest.fixture
def wavefile():
    return "tests/data/039_Cs_40_26.5_observe.dat"


def test_11B_wave(wavefile):
    data = read11B(wavefile, "wave")[1]
    s = get_shape(data)
    assert len(set(map(lambda x: x[0], s.values()))) == 1
    assert s["header"][0] == 46406


@pytest.fixture
def hkfile():
    return "tests/data/003_hk_90_ch0.dat"


def assert_first_shape(s):
    for k, v in s.items():
        s[k] = v[0]
    assert len(set(s.values())) == 1


def test_11B_hk(hkfile):
    data = read11B(hkfile, "hk")[1]
    s = get_shape(data)
    assert s["header"][0] == 174
    assert_first_shape(s)
