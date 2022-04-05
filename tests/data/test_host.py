import pytest
import whois
from pythonping import ping
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
def test_host_registrar(url, expected):
    response = get_host(url)
    assert host_server_count(response) == expected

@pytest.mark.parametrize("url,date_type,date_attr,expected", [
    ("example.com", "creation_date", "day", 14),
    ("angkjnf.com", "creation_date", "day", None),
])
def test_host_date(url, date_type, date_attr, expected):
    response = get_host(url)
    assert host_date(response, date_type, date_attr) == expected

@pytest.mark.parametrize("url,expected", [
    ("example.com", 0.1),
    ("angkjnf.com", -1),
])
def test_host_speed(url, expected):
    assert host_speed(url) >= expected

@pytest.mark.parametrize("url,expected", [
    ("example.com", 69),
    ("angkjnf.com", -1),
])
def test_host_latency(url, expected):
    assert host_latency(url) >= expected