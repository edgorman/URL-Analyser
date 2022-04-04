

def content_type(response):
    try:
        return response.headers['Content-Type']
    except:
        return "NA"

def content_redirect(obj):
    try:
        return obj.is_redirect
    except:
        return None

def content_content(obj):
    try:
        return obj.content
    except:
        return "NA"
