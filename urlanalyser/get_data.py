"""
    get_data.py - Edward Gorman - eg6g17@soton.ac.uk
"""
import time
import whois
import requests
import pandas as pd
from datetime import datetime
from pythonping import ping
from urlanalyser import *


# Return valid url or None
def valid_url(url):
    if get_content(url) is not None:
        return url
    elif get_content("http://" + url) is not None:
        return "http://" + url
    elif get_content("https://" + url) is not None:
        return "https://" + url
    else:
        return None


# Get url host data
def get_host(url):
    try:
        return whois.whois(url)
    except:
        return None


# Get registrar of host
def get_host_registrar(obj, typ):
    try:
        return obj[typ]
    except:
        return None


# Get country of host
def get_host_country(obj, typ):
    try:
        return obj[typ]
    except:
        return None


# Get server count of host
def get_host_server(obj, typ):
    try:
        return len(obj[typ])
    except:
        return 0


# Get date of host
def get_host_date(obj, typ, attr_str):
    try:
        d = obj[typ]
    except:
        return None

    if isinstance(obj[typ], list):
        obj[typ] = obj[typ][0]

    if not isinstance(obj[typ], datetime):
        return None

    return getattr(obj[typ], attr_str)


# Get speed of host
def get_host_speed(url):
    t1 = time.perf_counter()
    host = get_host(url)
    return time.perf_counter() - t1


# Get latency of host
def get_host_latency(url):
    response = ping("google.com", size=40, count=10, timeout=1)
    return response.rtt_avg_ms


# Get url webpage content
def get_content(url):
    try:
        return requests.get(url, timeout=(0.1, 10))
    except:
        return None


# Get type of content
def get_content_type(response):
    try:
        return response.headers['Content-Type']
    except:
        return "NA"


# Get if content is redirect
def get_content_redirect(obj):
    try:
        return obj.is_redirect
    except:
        return None


# Get content of content
def get_content_content(obj):
    try:
        return obj.content
    except:
        return "NA"


def get_urls(df, limit):
    url_count = 0
    total_count = 0
    data = {'url': [],  # 'label': [],
            'location': [], 'registrar': [], 'servercount': [], 'creation_m': [], 'creation_y': [], 'updated_m': [], 'updated_y': [], 'expiration_m': [], 'expiration_y': [], 'latency': [],
            'redirect': [], 'type': [], 'length': [], 'content': []}

    # Iterate through all urls
    for index, row in df.iterrows():
        total_count = total_count + 1

        # Check if valid url
        url = valid_url(row['url'])
        if url is None:
            # Skip this url
            continue

        # Get url host and content
        host = get_host(url)
        content = get_content(url)

        # Add url to list
        data['url'].append(url)
        # data['label'].append(row['label'])
        data['location'].append(get_host_country(host, 'country'))
        data['registrar'].append(get_host_registrar(host, 'registrar'))
        data['servercount'].append(get_host_server(host, 'name_servers'))
        data['creation_m'].append(get_host_date(host, 'creation_date', 'month'))
        data['creation_y'].append(get_host_date(host, 'creation_date', 'year'))
        data['updated_m'].append(get_host_date(host, 'updated_date', 'month'))
        data['updated_y'].append(get_host_date(host, 'updated_date', 'year'))
        data['expiration_m'].append(get_host_date(host, 'expiration_date', 'month'))
        data['expiration_y'].append(get_host_date(host, 'expiration_date', 'year'))
        data['latency'].append(get_host_latency(url))
        data['redirect'].append(get_content_redirect(content))
        data['type'].append(get_content_type(content))
        data['length'].append(len(get_content_content(content)))
        data['content'].append(get_content_content(content)[0:100000])
        url_count = url_count + 1
        # print(url_count, "/", total_count)

        if url_count >= limit:
            break

    return pd.DataFrame(data=data)


if __name__ == "__main__":
    # Load lexical data
    print("Loading data . . .")
    df = pd.read_csv(DATA_DIRECTORY + LEXICAL_FILE, sep="\t")

    # Get web content
    print("Getting malicious urls . . .")
    mal_df = get_urls(df[df['label'] == 1], 10000)
    print("Getting benign urls . . .")
    ben_df = get_urls(df[df['label'] == 0], 10000)

    # Joining malicious and benign dataframes
    df = pd.concat([mal_df, ben_df])
    df = df.reset_index()

    # Splitting into host and content features
    print("Outputting to files . . .")
    host_df = df[['url', 'label', 'location', 'registrar', 'servercount', 'creation_m', 'creation_y', 'updated_m', 'updated_y', 'expiration_m', 'expiration_y', 'speed']]
    cont_df = df[['url', 'label', 'redirect', 'type', 'length', 'content']]

    # Saving to files
    host_df.to_csv(path_or_buf=DATA_DIRECTORY + HOST_FILE, sep="\t")
    cont_df.to_csv(path_or_buf=DATA_DIRECTORY + CONTENT_FILE, sep="\t")
