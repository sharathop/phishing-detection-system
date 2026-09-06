"""
feature_extraction.py  –  PhishGuard
Extracts the 56 lexical features used by the XGBoost model.

All features are derived purely from URL structure — no network calls,
no page fetches, no WHOIS. This matches exactly what the model was
trained on, preventing train/inference distribution mismatch.

Usage:
    from feature_extraction import extract_features
    features = extract_features("https://example.com/login?id=1")
    # returns list of 56 numeric values
"""

import re
import socket
import tldextract
from urllib.parse import urlparse

# ── Known phishing-hint keywords ─────────────────────────────
PHISH_HINTS = [
    'login', 'verify', 'secure', 'update', 'confirm', 'account',
    'password', 'billing', 'suspended', 'recover', 'alert', 'unlock',
    'validate', 'authenticate', 'authorize', 'reactivate', 'restore',
    'renew', 'webscr', 'signin', 'banking', 'ebayisapi', 'paypal'
]

SHORTENING_SERVICES = [
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
    'buff.ly', 'adf.ly', 'bit.do', 'mcaf.ee', 'su.pr'
]

SUSPICIOUS_TLDS = [
    '.zip', '.review', '.country', '.kim', '.cricket', '.science',
    '.work', '.party', '.gq', '.link', '.tk', '.ml', '.cf', '.ga'
]

BRANDS = [
    'google', 'facebook', 'apple', 'microsoft', 'amazon', 'paypal',
    'netflix', 'instagram', 'twitter', 'linkedin', 'dropbox', 'adobe',
    'ebay', 'yahoo', 'bing', 'outlook', 'office', 'whatsapp'
]

# ── Helpers ───────────────────────────────────────────────────

def _count(string, char):
    return string.count(char)

def _ratio_digits(s):
    if not s:
        return 0.0
    return sum(c.isdigit() for c in s) / len(s)

def _is_ip(hostname):
    try:
        socket.inet_aton(hostname)
        return 1
    except Exception:
        pass
    if re.match(r'^\[?[0-9a-fA-F:]+\]?$', hostname):
        return 1
    return 0

def _has_punycode(hostname):
    return 1 if 'xn--' in hostname.lower() else 0

def _get_port(parsed):
    return 1 if parsed.port and parsed.port not in (80, 443) else 0

def _abnormal_subdomain(subdomain):
    if not subdomain:
        return 0
    for p in subdomain.split('.'):
        if re.search(r'\d', p) and re.search(r'[a-zA-Z]', p):
            return 1
    return 0

def _random_domain(domain):
    """High consonant ratio → likely random/generated domain."""
    if not domain:
        return 0
    vowels = sum(1 for c in domain.lower() if c in 'aeiou')
    return 1 if (vowels / len(domain)) < 0.2 else 0

def _word_stats(text):
    """Returns (num_words, char_repeat, shortest, longest, avg_len)."""
    words = re.split(r'[\W_]+', text)
    words = [w for w in words if w]
    if not words:
        return 0, 0, 0, 0, 0.0
    lengths = [len(w) for w in words]
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    char_repeat = max(freq.values()) if freq else 0
    return (
        len(words),
        char_repeat,
        min(lengths),
        max(lengths),
        sum(lengths) / len(lengths)
    )


# ── Main extractor ────────────────────────────────────────────

def extract_features(url: str) -> list:
    """
    Returns a list of 56 floats/ints matching the 56 lexical training columns.
    Column order matches LEXICAL_COLS in model.py exactly.
    """
    parsed    = urlparse(url)
    ext       = tldextract.extract(url)
    hostname  = parsed.netloc or ''
    domain    = ext.domain or ''
    sub       = ext.subdomain or ''
    tld       = ('.' + ext.suffix) if ext.suffix else ''
    path      = parsed.path or ''
    full_url  = url

    # ── 1. length_url
    f1  = len(full_url)
    # ── 2. length_hostname
    f2  = len(hostname)
    # ── 3. ip
    f3  = _is_ip(hostname.replace('www.', ''))
    # ── 4. nb_dots
    f4  = _count(full_url, '.')
    # ── 5. nb_hyphens
    f5  = _count(full_url, '-')
    # ── 6. nb_at
    f6  = _count(full_url, '@')
    # ── 7. nb_qm
    f7  = _count(full_url, '?')
    # ── 8. nb_and
    f8  = _count(full_url, '&')
    # ── 9. nb_or
    f9  = _count(full_url, '|')
    # ── 10. nb_eq
    f10 = _count(full_url, '=')
    # ── 11. nb_underscore
    f11 = _count(full_url, '_')
    # ── 12. nb_tilde
    f12 = _count(full_url, '~')
    # ── 13. nb_percent
    f13 = _count(full_url, '%')
    # ── 14. nb_slash
    f14 = _count(full_url, '/')
    # ── 15. nb_star
    f15 = _count(full_url, '*')
    # ── 16. nb_colon
    f16 = _count(full_url, ':')
    # ── 17. nb_comma
    f17 = _count(full_url, ',')
    # ── 18. nb_semicolumn
    f18 = _count(full_url, ';')
    # ── 19. nb_dollar
    f19 = _count(full_url, '$')
    # ── 20. nb_space
    f20 = _count(full_url, ' ') + _count(full_url, '%20')
    # ── 21. nb_www
    f21 = full_url.lower().count('www')
    # ── 22. nb_com
    f22 = full_url.lower().count('.com')
    # ── 23. nb_dslash
    f23 = full_url.count('//')
    # ── 24. http_in_path
    f24 = 1 if 'http' in path.lower() else 0
    # ── 25. https_token
    # NOTE: matches training-data semantics exactly (verified against
    # dataset_phishing.csv): this column is 1 for http:// URLs, 0 for https://.
    # It is effectively "is_not_https", not "is_https" — do not flip this.
    f25 = 1 if parsed.scheme != 'https' else 0
    # ── 26. ratio_digits_url
    f26 = _ratio_digits(full_url)
    # ── 27. ratio_digits_host
    f27 = _ratio_digits(hostname)
    # ── 28. punycode
    f28 = _has_punycode(hostname)
    # ── 29. port
    f29 = _get_port(parsed)
    # ── 30. tld_in_path
    f30 = 1 if tld and tld in path.lower() else 0
    # ── 31. tld_in_subdomain
    f31 = 1 if tld and tld in sub.lower() else 0
    # ── 32. abnormal_subdomain
    f32 = _abnormal_subdomain(sub)
    # ── 33. nb_subdomains
    f33 = len(sub.split('.')) if sub else 0
    # ── 34. prefix_suffix
    f34 = 1 if '-' in domain else 0
    # ── 35. random_domain
    f35 = _random_domain(domain)
    # ── 36. shortening_service
    reg_domain = f"{ext.domain}.{ext.suffix}".lower() if ext.suffix else domain.lower()
    f36 = 1 if reg_domain in SHORTENING_SERVICES else 0
    # ── 37. path_extension
    ext_match = re.search(r'\.([a-zA-Z0-9]{2,5})$', path.split('?')[0])
    f37 = 1 if ext_match else 0
    # ── 38. nb_redirection (// after scheme)
    f38 = len(re.findall(r'(?<!:)//', full_url))
    # ── 39. nb_external_redirection
    f39 = full_url.lower().count('url=') + full_url.lower().count('redirect=')

    # ── Word stats ────────────────────────────────────────────
    url_words, url_char_rep, url_short, url_long, url_avg   = _word_stats(full_url)
    host_words, _, host_short, host_long, host_avg           = _word_stats(hostname)
    path_words, _, path_short, path_long, path_avg           = _word_stats(path)

    # ── 40. length_words_raw
    f40 = url_words
    # ── 41. char_repeat
    f41 = url_char_rep
    # ── 42. shortest_words_raw
    f42 = url_short
    # ── 43. shortest_word_host
    f43 = host_short
    # ── 44. shortest_word_path
    f44 = path_short
    # ── 45. longest_words_raw
    f45 = url_long
    # ── 46. longest_word_host
    f46 = host_long
    # ── 47. longest_word_path
    f47 = path_long
    # ── 48. avg_words_raw
    f48 = url_avg
    # ── 49. avg_word_host
    f49 = host_avg
    # ── 50. avg_word_path
    f50 = path_avg

    # ── 51. phish_hints
    f51 = sum(1 for hint in PHISH_HINTS if hint in full_url.lower())
    # ── 52. domain_in_brand
    f52 = 1 if any(b == domain.lower() for b in BRANDS) else 0
    # ── 53. brand_in_subdomain
    f53 = 1 if any(b in sub.lower() for b in BRANDS) else 0
    # ── 54. brand_in_path
    f54 = 1 if any(b in path.lower() for b in BRANDS) else 0
    # ── 55. suspecious_tld  (matches dataset column spelling exactly)
    f55 = 1 if any(full_url.lower().endswith(t) or tld == t for t in SUSPICIOUS_TLDS) else 0
    # ── 56. statistical_report
    f56 = 1 if f51 >= 2 and (f55 or f3) else 0

    features = [
        f1,  f2,  f3,  f4,  f5,  f6,  f7,  f8,  f9,  f10,
        f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
        f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
        f31, f32, f33, f34, f35, f36, f37, f38, f39, f40,
        f41, f42, f43, f44, f45, f46, f47, f48, f49, f50,
        f51, f52, f53, f54, f55, f56
    ]

    assert len(features) == 56, f"Feature count mismatch: {len(features)}"
    return features