from datetime import datetime
import pytest
import whois

from URLAnalyser.data.host import get_host
from URLAnalyser.data.host import host_registrar
from URLAnalyser.data.host import host_country
from URLAnalyser.data.host import host_server_count
from URLAnalyser.data.host import host_date
from URLAnalyser.data.host import host_speed
from URLAnalyser.data.host import host_latency


@pytest.mark.parametrize("url,expected", [
    ("example.com", whois.WhoisEntry),
    ("angkjnf.com", type(None)),
])
def test_get_host(url, expected):
    assert isinstance(get_host(url), expected)


@pytest.mark.parametrize("url,expected", [
    ("example.com", "RESERVED-Internet Assigned Numbers Authority"),
    ("angkjnf.com", None),
])
def test_host_registrar(url, expected):
    response = get_host(url)
    assert host_registrar(response) == expected


@pytest.mark.skip(reason="google.com seems to flip between 'US' and None for some reason")
@pytest.mark.parametrize("url,expected", [
    ("google.com", "US"),
    ("angkjnf.com", None),
])
def test_host_country(url, expected):
    response = get_host(url)
    assert host_country(response) == expected


@pytest.mark.parametrize("url,expected", [
    ("example.com", 2),
    ("angkjnf.com", 0),
])
def test_host_server_count(url, expected):
    response = get_host(url)
    assert host_server_count(response) == expected


@pytest.mark.parametrize("url,date_type,expected", [
    ("example.com", "creation_date", datetime(1995, 8, 14, 4, 0)),
    ("angkjnf.com", "creation_date", None),
])
def test_host_date(url, date_type, expected):
    response = get_host(url)
    assert host_date(response, date_type) == expected


@pytest.mark.parametrize("url,expected", [
    ("example.com", 0),
    ("angkjnf.com", -1),
])
def test_host_speed(url, expected):
    assert host_speed(url) >= expected


@pytest.mark.parametrize("url,expected", [
    ("google.com", 5),
    ("angkjnf.com", -1),
])
def test_host_latency(url, expected):
    assert host_latency(url) >= expected
