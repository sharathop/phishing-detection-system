from urllib.parse import urlparse

# ---------- FEATURE EXTRACTOR ----------
def extract_features(url):
    parsed = urlparse(url)

    features = [
        len(url),                                # URL length
        url.count('.'),                          # dots
        url.count('-'),                          # hyphens
        1 if parsed.scheme == "https" else 0,    # https
        1 if any(c.isdigit() for c in url) else 0,
        1 if any(w in url.lower() for w in
                 ['login','signin','verify','update','secure','account']) else 0,
        len(parsed.netloc),                      # domain length
        len(parsed.path)                         # path length
    ]

    return features