import pytest
from URLAnalyser.features.content import average_word_length
from URLAnalyser.features.content import average_js_length


@pytest.mark.parametrize("words,expected", [
    ("the quick brown fox jumped over the lazy dog", 36 / float(9)),
    ("lorem ipsum...", 13 / float(2)),
])
def test_average_word_length(words, expected):
    assert average_word_length(words.split(" ")) == expected


@pytest.mark.parametrize("words,expected", [
    ("the quick brown fox jumped over the lazy dog", 0),
    ("lorem ipsum...", 0),
    ("lorem <script>these are words</script> ipsum...", 13 / float(3)),
])
def test_average_js_length(words, expected):
    assert average_js_length(words.split(" ")) == expected
