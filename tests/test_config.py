from src.config import ROOT, REL_THRESHOLD, TEST_FRAC, VAL_FRAC


def test_root_exists():
    assert ROOT.exists()


def test_constants():
    assert REL_THRESHOLD == 4.0
    assert 0 < VAL_FRAC < 1 and 0 < TEST_FRAC < 1
