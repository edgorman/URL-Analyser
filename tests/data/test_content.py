import string
import pytest
import requests
from URLAnalyser.data.content import get_content
from URLAnalyser.data.content import content_type
from URLAnalyser.data.content import content_redirect
from URLAnalyser.data.content import content_content


@pytest.mark.parametrize("url,expected", [
    ("example.com", requests.models.Response),
    ("angkjnf.com", type(None)),
])
def test_get_content(url, expected):
    assert isinstance(get_content(url), expected)

@pytest.mark.parametrize("url,expected", [
    ("example.com", "text/html; charset=UTF-8"),
    ("angkjnf.com", None),
])
def test_content_type(url, expected):
    response = get_content(url)
    assert content_type(response) == expected

@pytest.mark.skip(reason="Can't find example URL that redirects")
@pytest.mark.parametrize("url,expected", [
    ("example.com", False),
    ("w.wiki/4ozb", True),
    ("angkjnf.com", None),
])
def test_content_redirect(url, expected):
    response = get_content(url)
    assert content_redirect(response) == expected

@pytest.mark.parametrize("url,expected", [
    ("example.com", str),
    ("angkjnf.com", type(None)),
])
def test_content_content(url, expected):
    response = get_content(url)
    assert isinstance(content_content(response), expected)
