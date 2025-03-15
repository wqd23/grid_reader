import pytest
from reader import read11B
from .test_A import get_shape


@pytest.fixture
def wavefile():
    return "tests/data/039_Cs_40_26.5_observe.dat"

def test_11B_wave(wavefile):
    data = read11B(wavefile, "wave")
    s = get_shape(data)
    assert len(set(map(lambda x:x[0], s.values()))) == 1

@pytest.fixture
def featfile():
    return "tests/data/088_observe_65_ch2.dat"

def test_11B_feat(featfile):
    data = read11B(featfile, "feat")
    s = get_shape(data)
    assert len(set(s.values())) == 1