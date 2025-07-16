import re

def extract_keywords(row):
    """
    Extract keywords from a row's title and tags (if present).
    Returns a list of keywords.
    """
    keywords = []
    title = str(row.get('title', ''))
    tags = row.get('tags', [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(',') if t.strip()]
    # Basic keyword extraction: split title, add tags
    keywords.extend(re.findall(r'\b\w+\b', title.lower()))
    keywords.extend([t.lower() for t in tags])
    return [k for k in keywords if len(k) > 2]
