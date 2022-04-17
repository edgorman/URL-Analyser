import pytest

from URLAnalyser.log import Log


@pytest.mark.parametrize("verboseness,expected", [
    (0, 0),
    (1, 20),
])
def test_info(verboseness, expected, tmp_out):
    Log.verboseness = verboseness
    Log.info("this is a test")

    assert len(tmp_out.getvalue()) >= expected


@pytest.mark.parametrize("verboseness,expected", [
    (0, 0),
    (1, 20),
])
def test_success(verboseness, expected, tmp_out):
    Log.verboseness = verboseness
    Log.success("this is a test")

    assert len(tmp_out.getvalue()) >= expected


@pytest.mark.parametrize("verboseness,expected", [
    (0, 20),
    (1, 20),
])
def test_error(verboseness, expected, tmp_out):
    Log.verboseness = verboseness
    with pytest.raises(SystemExit) as e:
        Log.error("this is a test")

    assert len(tmp_out.getvalue()) >= expected
    assert e.type == SystemExit
    assert e.value.code == -1


@pytest.mark.parametrize("verboseness,expected", [
    (0, 20),
    (1, 20),
])
def test_result(verboseness, expected, tmp_out):
    Log.verboseness = verboseness
    Log.result("this is a test")

    assert len(tmp_out.getvalue()) >= expected
