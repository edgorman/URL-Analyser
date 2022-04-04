import time
import datetime

import whois
from pythonping import ping


def get_host(url):
    try:
        return whois.whois(url)
    except:
        return None

def host_registrar(obj, typ):
    try:
        return obj[typ]
    except:
        return None

def host_country(obj, typ):
    try:
        return obj[typ]
    except:
        return None

def host_server(obj, typ):
    try:
        return len(obj[typ])
    except:
        return 0

def host_date(obj, typ, attr_str):
    try:
        d = obj[typ]
    except:
        return None

    if isinstance(obj[typ], list):
        obj[typ] = obj[typ][0]

    if not isinstance(obj[typ], datetime):
        return None

    return getattr(obj[typ], attr_str)

def host_speed(url):
    t1 = time.perf_counter()
    _ = get_host(url)
    return time.perf_counter() - t1

def host_latency(url):
    response = ping("google.com", size=40, count=10, timeout=1)
    return response.rtt_avg_ms
