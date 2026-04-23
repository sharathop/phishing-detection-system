"""
feature_extraction.py  –  PhishGuard
Extracts all 86 features in the exact order expected by the XGBoost model.

Lexical features (fast, no network):  cols 0–55
External features (network/WHOIS):    cols 56–85  → default safe values used
                                       if lookup fails or times out.

Usage:
    from feature_extraction import extract_features
    features = extract_features("https://example.com/login?id=1")
    # returns list of 86 numeric values
"""

import re
import math
import socket
import requests
import tldextract
import whois
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from bs4 import BeautifulSoup

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
        return 0
    return sum(c.isdigit() for c in s) / len(s)

def _is_ip(hostname):
    try:
        socket.inet_aton(hostname)
        return 1
    except Exception:
        pass
    # IPv6
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
    parts = subdomain.split('.')
    for p in parts:
        if re.search(r'\d', p) and re.search(r'[a-zA-Z]', p):
            return 1
    return 0

def _random_domain(domain):
    """Rough entropy check – high entropy → likely random."""
    if not domain:
        return 0
    vowels = sum(1 for c in domain.lower() if c in 'aeiou')
    ratio = vowels / len(domain)
    return 1 if ratio < 0.2 else 0

def _word_stats(text):
    """Returns (length_words_raw, char_repeat, shortest, longest, avg)."""
    words = re.split(r'[\W_]+', text)
    words = [w for w in words if w]
    if not words:
        return 0, 0, 0, 0, 0.0
    lengths = [len(w) for w in words]
    # char_repeat: max frequency of any single char
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

def _fetch_page(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout, allow_redirects=True,
                         headers={'User-Agent': 'Mozilla/5.0'})
        return r
    except Exception:
        return None

def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


# ── Main extractor ────────────────────────────────────────────

def extract_features(url: str) -> list:
    """
    Returns a list of 86 floats/ints matching the training dataset columns.
    External lookups that fail silently return safe neutral defaults.
    """
    parsed      = urlparse(url)
    ext         = tldextract.extract(url)
    hostname    = parsed.netloc or ''
    domain      = ext.domain or ''
    subdomain   = ext.suffix or ''
    tld         = ('.' + ext.suffix) if ext.suffix else ''
    path        = parsed.path or ''
    query       = parsed.query or ''
    full_url    = url

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
    f25 = 1 if parsed.scheme == 'https' else 0
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
    sub = ext.subdomain or ''
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
    f36 = 1 if any(s in full_url.lower() for s in SHORTENING_SERVICES) else 0
    # ── 37. path_extension
    ext_match = re.search(r'\.([a-zA-Z0-9]{2,5})$', path.split('?')[0])
    f37 = 1 if ext_match else 0
    # ── 38. nb_redirection (// after scheme)
    f38 = len(re.findall(r'(?<!:)//', full_url))
    # ── 39. nb_external_redirection
    f39 = full_url.lower().count('url=') + full_url.lower().count('redirect=')

    # ── Word stats on full URL ────────────────────────────────
    url_words, url_char_rep, url_short, url_long, url_avg = _word_stats(full_url)
    host_words, _, host_short, host_long, host_avg         = _word_stats(hostname)
    path_words, _, path_short, path_long, path_avg         = _word_stats(path)

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
    # ── 55. suspicious_tld
    f55 = 1 if any(full_url.lower().endswith(t) or tld == t for t in SUSPICIOUS_TLDS) else 0
    # ── 56. statistical_report (heuristic: flag if in known phishing TLD + hints)
    f56 = 1 if f51 >= 2 and (f55 or f3) else 0

    # ── External / page-content features (57–86) ─────────────
    # Defaults (safe-side neutral values used on failure)
    f57  = 0   # nb_hyperlinks
    f58  = 0.0 # ratio_intHyperlinks
    f59  = 0.0 # ratio_extHyperlinks
    f60  = 0.0 # ratio_nullHyperlinks
    f61  = 0   # nb_extCSS
    f62  = 0.0 # ratio_intRedirection
    f63  = 0.0 # ratio_extRedirection
    f64  = 0.0 # ratio_intErrors
    f65  = 0.0 # ratio_extErrors
    f66  = 0   # login_form
    f67  = 0   # external_favicon
    f68  = 0.0 # links_in_tags
    f69  = 0   # submit_email
    f70  = 0.0 # ratio_intMedia
    f71  = 0.0 # ratio_extMedia
    f72  = 0   # sfh (server form handler)
    f73  = 0   # iframe
    f74  = 0   # popup_window
    f75  = 0.0 # safe_anchor
    f76  = 0   # onmouseover
    f77  = 0   # right_clic
    f78  = 0   # empty_title
    f79  = 1   # domain_in_title  (default: assume present = good)
    f80  = 0   # domain_with_copyright
    f81  = 0   # whois_registered_domain
    f82  = 0   # domain_registration_length
    f83  = 0   # domain_age
    f84  = 0   # web_traffic
    f85  = 0   # dns_record
    f86  = 0   # google_index
    f87  = 0   # page_rank  (feature 86 in 0-indexed = index 85)

    try:
        response = _fetch_page(url)
        if response is not None:
            soup = BeautifulSoup(response.text, 'html.parser')
            base = f"{parsed.scheme}://{parsed.netloc}"

            # Hyperlinks
            anchors   = soup.find_all('a', href=True)
            all_links = [a['href'] for a in anchors]
            nb_links  = len(all_links)
            f57 = nb_links

            if nb_links:
                int_links  = [l for l in all_links if base in l or l.startswith('/')]
                ext_links  = [l for l in all_links if base not in l and l.startswith('http')]
                null_links = [l for l in all_links if l in ('#', '', 'javascript:void(0)')]
                f58 = len(int_links) / nb_links
                f59 = len(ext_links) / nb_links
                f60 = len(null_links) / nb_links

            # External CSS
            css_links = soup.find_all('link', rel=lambda r: r and 'stylesheet' in r)
            f61 = sum(1 for c in css_links if c.get('href', '').startswith('http') and base not in c.get('href', ''))

            # Forms
            forms = soup.find_all('form')
            if forms:
                actions = [f.get('action', '') for f in forms]
                int_forms = [a for a in actions if not a.startswith('http') or base in a]
                ext_forms = [a for a in actions if a.startswith('http') and base not in a]
                if actions:
                    f62 = len(int_forms) / len(actions)
                    f63 = len(ext_forms) / len(actions)
                f66 = 1 if any(re.search(r'(login|signin|password|credential)', str(f), re.I) for f in forms) else 0
                f72 = 1 if any(a.startswith('mailto:') or a == '' for a in actions) else 0

            # Favicon
            favicon = soup.find('link', rel=lambda r: r and 'icon' in r)
            if favicon and favicon.get('href', '').startswith('http'):
                f67 = 1 if base not in favicon['href'] else 0

            # Links in tags (script/link/img)
            tag_links = soup.find_all(['script', 'link', 'img'])
            nb_tag    = len(tag_links)
            if nb_tag:
                int_tag = [t for t in tag_links if base in (t.get('src', '') or t.get('href', ''))]
                f68 = len(int_tag) / nb_tag

            # submit_email
            f69 = 1 if soup.find('input', {'type': 'email'}) else 0

            # Media
            media = soup.find_all(['img', 'video', 'audio'])
            nb_media = len(media)
            if nb_media:
                int_media = [m for m in media if not (m.get('src', '') or '').startswith('http') or base in (m.get('src', '') or '')]
                ext_media = [m for m in media if (m.get('src', '') or '').startswith('http') and base not in (m.get('src', '') or '')]
                f70 = len(int_media) / nb_media
                f71 = len(ext_media) / nb_media

            # iframe
            f73 = 1 if soup.find('iframe') else 0

            # popup / onmouseover / right-click disable
            page_text = response.text
            f74 = 1 if 'window.open(' in page_text else 0
            f75 = 1 - f59  # safe_anchor ≈ inverse of ext ratio
            f76 = 1 if 'onmouseover' in page_text.lower() else 0
            f77 = 1 if 'contextmenu' in page_text.lower() and 'return false' in page_text.lower() else 0

            # Title checks
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.get_text().strip()
                f78 = 1 if not title_text else 0
                f79 = 1 if domain.lower() in title_text.lower() else 0
            else:
                f78 = 1
                f79 = 0

            # Copyright
            f80 = 1 if (domain.lower() in page_text.lower() and '©' in page_text) else 0

    except Exception:
        pass  # keep defaults

    # ── WHOIS / DNS features ──────────────────────────────────
    try:
        w = whois.whois(hostname)
        if w.domain_name:
            f81 = 1
            # registration length (years)
            exp  = w.expiration_date
            cre  = w.creation_date
            if isinstance(exp, list): exp = exp[0]
            if isinstance(cre, list): cre = cre[0]
            if exp and cre:
                f82 = max(0, (exp - cre).days // 365)
            # domain age (days since creation)
            if cre:
                f83 = (datetime.now() - cre).days
    except Exception:
        pass

    # ── DNS record ───────────────────────────────────────────
    try:
        socket.getaddrinfo(hostname, None)
        f85 = 1
    except Exception:
        f85 = 0

    features = [
        f1,  f2,  f3,  f4,  f5,  f6,  f7,  f8,  f9,  f10,
        f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
        f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
        f31, f32, f33, f34, f35, f36, f37, f38, f39, f40,
        f41, f42, f43, f44, f45, f46, f47, f48, f49, f50,
        f51, f52, f53, f54, f55, f56,
        f57, f58, f59, f60, f61, f62, f63, f64, f65, f66,
        f67, f68, f69, f70, f71, f72, f73, f74, f75, f76,
        f77, f78, f79, f80, f81, f82, f83, f84, f85, f86
    ]

    # Sanity check
    assert len(features) == 86, f"Feature count mismatch: {len(features)}"
    return features