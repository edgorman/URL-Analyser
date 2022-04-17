import time
import datetime

import whois
from pythonping import ping


def get_host(url: str) -> whois.WhoisEntry:
    try:
        return whois.whois(url)
    except BaseException:
        return None


def host_registrar(response: whois.WhoisEntry) -> str:
    return response['registrar'] if response is not None else None


def host_country(response: whois.WhoisEntry) -> str:
    return response['country'] if response is not None and 'country' in response else None


def host_server_count(response: whois.WhoisEntry) -> int:
    return len(response['name_servers']) if response is not None and 'country' in response else 0


def host_date(response: whois.WhoisEntry, date_type: str) -> str:
    if response is None or date_type not in response:
        return None

    if isinstance(response[date_type], list):
        response[date_type] = response[date_type][0]

    if not isinstance(response[date_type], datetime.datetime):
        return None

    return response[date_type]


def host_speed(url: str) -> float:
    t1 = time.perf_counter()
    response = get_host(url)
    t2 = time.perf_counter()

    return t2 - t1 if response is not None else -1


def host_latency(url: str) -> float:
    try:
        return ping(url, count=10, timeout=1).rtt_avg_ms
    except BaseException:
        return -1
