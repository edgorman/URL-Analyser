import string
import pytest
from URLAnalyser.features.lexical import label_count
from URLAnalyser.features.lexical import average_label_length
from URLAnalyser.features.lexical import char_count


@pytest.mark.parametrize("url,expected", [
    ("google.com", 2),
    ("edgorman.github.io", 3),
    ("longer.url/with?path=value", 2)
])
def test_label_count(url, expected):
    assert label_count(url) == expected

@pytest.mark.parametrize("url,expected", [
    ("google.com", 9/float(2)),
    ("edgorman.github.io", 16/float(3)),
    ("longer.url/with?path=value", 25/float(2))
])
def test_label_count(url, expected):
    assert average_label_length(url) == expected

@pytest.mark.parametrize("url,list,expected", [
    ("abcdefg.com", string.ascii_letters, 10),
    ("()()()().com", string.ascii_letters, 3),
    ("abcdefg.com", './?=-_', 1),
    ("()()()().com?path=---", './?=-_', 6),
    ("abcdefg.com", string.digits, 0),
    ("()()()().com?path=117", string.digits, 3),
])
def test_char_count(url, list, expected):
    assert char_count(url, list) == expected
