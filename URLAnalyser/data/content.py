import requests


def get_content(url: str) -> requests.models.Response:
    try:
        return requests.get("http://" + url, timeout=(0.1, 10))
    except BaseException:
        try:
            return requests.get("https://" + url, timeout=(0.1, 10))
        except BaseException:
            return None


def content_type(response: requests.models.Response) -> str:
    return response.headers['Content-Type'] if response is not None else None


def content_redirect(response: requests.models.Response) -> bool:
    return response.is_redirect if response is not None else None


def content_content(response: requests.models.Response) -> str:
    return response.content.decode(response.apparent_encoding)[:10000] if response is not None else None
