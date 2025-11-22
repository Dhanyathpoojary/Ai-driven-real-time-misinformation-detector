# app.py
from flask import Flask, render_template, request
import requests
import re
import random
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from newspaper import Article
from sentence_transformers import SentenceTransformer, util
from gnews import GNews
from functools import lru_cache
from collections import defaultdict
from datetime import datetime

app = Flask(__name__)

# ==================================================
# CONFIG
# ==================================================
NEWS_API_KEY = "PUT YOUR API HERE"
NEWSAPI_COUNTRY = "in"
NEWSAPI_LANG = "en"
RESULTS_PER_PAGE = 10

# SentenceTransformer model (same as your original)
model = SentenceTransformer("all-MiniLM-L6-v2")

TRUSTED_DOMAINS = {
    "thehindu.com", "indianexpress.com", "hindustantimes.com",
    "ndtv.com", "timesofindia.indiatimes.com", "economictimes.indiatimes.com",
    "livemint.com", "moneycontrol.com", "scroll.in", "theprint.in",
    "news18.com", "business-standard.com", "firstpost.com", "pib.gov.in",
    "deccanherald.com", "thewire.in", "tribuneindia.com", "dnaindia.com",
    "zeenews.india.com", "indiatoday.in", "newindianexpress.com"
}

REAL_SIM, UNSURE_SIM, TITLE_MATCH_SIM = 0.60, 0.35, 0.60

DEBUNK_KEYWORDS = [
    "fake", "false", "fact check", "fact-check", "misleading", "debunk",
    "hoax", "rumour", "rumor", "not true", "no,", "claims are false",
    "fabricated", "bogus", "myth", "morphed"
]

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def hostname(url):
    try:
        return url.split("/")[2]
    except:
        return "Unknown"

def base_domain(url):
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except:
        return ""

def normalize_text(s):
    return re.sub(r"\s+", " ", s.strip()) if s else ""

def is_trusted(url):
    dom = base_domain(url)
    return any(dom == t or dom.endswith("." + t) for t in TRUSTED_DOMAINS)

def contains_debunk_lang(text):
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in DEBUNK_KEYWORDS)

# ==================================================
# TEXT EXTRACTION
# ==================================================
@lru_cache(maxsize=512)
def _extract_with_newspaper(url):
    try:
        art = Article(url)
        art.download()
        art.parse()
        return art.text[:10000] if art.text else "", art.top_image
    except:
        return "", None

def extract_text_and_image(url):
    text, img = "", None
    t, i = _extract_with_newspaper(url)
    if t:
        text = t
    if i:
        img = i

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        if not text:
            ps = soup.find_all("p")
            text = " ".join(p.get_text(" ", strip=True) for p in ps)[:10000]
        if not img:
            og = soup.find("meta", property="og:image")
            if og and og.get("content"):
                img = og["content"]
        if not img:
            tag = soup.find("img")
            if tag and tag.get("src"):
                img = urljoin(url, tag.get("src"))
    except:
        pass

    if not img:
        img = "/static/images/news_placeholder.jpg"

    return normalize_text(text), img, hostname(url)

# ==================================================
# NEWS FETCHERS
# ==================================================
def parse_newsapi(raw):
    return [{
        "title": a.get("title"),
        "url": a.get("url"),
        "image": a.get("urlToImage") or "/static/images/news_placeholder.jpg",
        "source": a.get("source", {}).get("name", hostname(a.get("url") or "")) or "Unknown"
    } for a in raw]

def fetch_newsapi(q="", page_size=25):
    try:
        params = {
            "q": q or "india",
            "pageSize": page_size,
            "language": NEWSAPI_LANG,
            "sortBy": "publishedAt",
            "apiKey": NEWS_API_KEY
        }
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        return parse_newsapi(r.json().get("articles", []))
    except:
        return []

gn = GNews(language="en", country="IN", max_results=20)

def fetch_gnews(q=""):
    try:
        items = gn.get_news(q or "India")
        return [{
            "title": n.get("title"),
            "url": n.get("url"),
            "image": n.get("image") or "/static/images/news_placeholder.jpg",
            "source": hostname(n.get("url") or "")
        } for n in items]
    except:
        return []

def aggregate_candidates_from_keyword(q):
    seen = set()
    items = []
    for src in (fetch_newsapi(q), fetch_gnews(q)):
        for a in src:
            u = a.get("url")
            if not u or u in seen:
                continue
            seen.add(u)
            items.append(a)
    return items

def fetch_top_headlines():
    seen = set()
    news = []
    for src in (fetch_newsapi("", 15), fetch_gnews("latest India news")):
        for n in src:
            if n["url"] not in seen:
                seen.add(n["url"])
                news.append(n)
    return news

# ==================================================
# SIMILARITY & VERDICT
# ==================================================
@lru_cache(maxsize=2048)
def _enc(txt):
    return model.encode(txt, convert_to_tensor=True)

def sim_headline_body(h, b):
    return float(util.cos_sim(_enc(normalize_text(h)), _enc(normalize_text(b)))) if h and b else 0.0

def detect_debunk_in_article(url):
    if not is_trusted(url):
        return False
    try:
        body, _, _ = extract_text_and_image(url)
        return contains_debunk_lang(body)
    except:
        return False

def presence_based_verdict(headline, article):
    url = article.get("url")
    body, img, _ = extract_text_and_image(url)
    if detect_debunk_in_article(url):
        return "FAKE", 97.0, img, body

    sim = sim_headline_body(headline or article.get("title"), body)
    base = 88 if is_trusted(url) else 78
    bonus = 8 if sim >= REAL_SIM else (3 if sim >= UNSURE_SIM else 0)
    conf = min(99, round(base + bonus + random.uniform(-10, 10), 2))

    if conf < 60:
        status = "FAKE"
    elif conf < 80:
        status = "UNSURE"
    else:
        status = "REAL"

    return status, conf, img, body

# ==================================================
# ANALYTICS + HEATMAP (UPDATED to track all 3 traffic types)
# ==================================================
# Track per-country counts for each status
COUNTRY_COUNTS = defaultdict(lambda: {"REAL": 0, "UNSURE": 0, "FAKE": 0})
# Track per-source counts for each status
SOURCE_ACTIVITY = defaultdict(lambda: {"REAL": 0, "FAKE": 0, "UNSURE": 0})

COUNTRY_COORDS = {
    "India": [20.5937, 78.9629],
    "United States": [37.0902, -95.7129],
    "United Kingdom": [55.3781, -3.4360],
    "Canada": [56.1304, -106.3468],
    "Australia": [-25.2744, 133.7751],
    "Pakistan": [30.3753, 69.3451],
    "Bangladesh": [23.6850, 90.3563],
    "Sri Lanka": [7.8731, 80.7718],
    "Nepal": [28.3949, 84.1240],
    "Qatar": [25.276987, 51.520008],
    "United Arab Emirates": [23.4241, 53.8478],
    "Singapore": [1.3521, 103.8198],
    "International": [0, 0]
}

def _country_for_domain(dom):
    # simple heuristic: if domain ends with '.in' or contains '.in' treat as India
    if not dom:
        return "International"
    if dom.endswith(".in") or ".in" in dom:
        return "India"
    # fallback - you can expand mapping here
    return "International"

def record_activity(status, url):
    dom = base_domain(url or "")
    if not dom:
        dom = "International"

    # Normalize status keys
    status_key = status.upper() if isinstance(status, str) else str(status)

    # Log per-source
    SOURCE_ACTIVITY[dom][status_key] += 1

    # Map to country and increment per-status counters
    country = _country_for_domain(dom)
    COUNTRY_COUNTS[country][status_key] += 1

# ==================================================
# ANALYTICS ROUTE
# ==================================================
@app.route("/analytics")
def analytics():
    # prepare source_activity copy
    source_activity = {k: dict(v) for k, v in SOURCE_ACTIVITY.items()}

    # totals
    real_total = sum(v["REAL"] for v in source_activity.values())
    unsure_total = sum(v["UNSURE"] for v in source_activity.values())
    fake_total = sum(v["FAKE"] for v in source_activity.values())

    # Build heat_points -> intensity using weighted sum so heatmap can visualize combined traffic
    heat_points = []
    for country, counts in COUNTRY_COUNTS.items():
        if country in COUNTRY_COORDS:
            lat, lon = COUNTRY_COORDS[country]
            # weight: FAKE=1.0, UNSURE=0.6, REAL=0.3 (so FAKE appears hotter)
            intensity = counts.get("FAKE", 0) * 1.0 + counts.get("UNSURE", 0) * 0.6 + counts.get("REAL", 0) * 0.3
            if intensity > 0:
                heat_points.append([lat, lon, intensity])

    return render_template(
        "analytics.html",
        country_counts=dict(COUNTRY_COUNTS),
        source_activity=source_activity,
        heat_points=heat_points,
        real_total=real_total,
        unsure_total=unsure_total,
        fake_total=fake_total
    )

# ==================================================
# ROUTES (Home/Latest/Trending: now log REAL traffic)
# ==================================================
@app.route("/")
def index():
    news = fetch_top_headlines()
    # Log browsing activity as REAL
    for n in news:
        try:
            record_activity("REAL", n.get("url"))
        except:
            pass
    return render_template("home.html", news=news)

@app.route("/latest")
def latest():
    news = fetch_top_headlines()
    for n in news:
        try:
            record_activity("REAL", n.get("url"))
        except:
            pass
    return render_template("latest.html", news=news)

@app.route("/trending")
def trending():
    news = fetch_top_headlines()
    for n in news:
        try:
            record_activity("REAL", n.get("url"))
        except:
            pass
    return render_template("trending.html", news=news)

# ==================================================
# FACTCHECKS (keyword/url modes) — unchanged decision logic
# ==================================================
@app.route("/factchecks", methods=["GET", "POST"])
def factchecks():
    result = []
    query = ""
    mode = request.form.get("mode") or "keyword"

    if request.method == "POST":
        query = normalize_text(request.form.get("text"))

        if mode == "keyword":
            items = aggregate_candidates_from_keyword(query)
            for art in items:
                status, conf, img, body = presence_based_verdict(query, art)
                try:
                    record_activity(status, art.get("url"))
                except:
                    pass
                result.append(make_result(art, status, conf, img, body))

        elif mode == "url":
            url = query
            headline = get_title_from_url(url)
            art = {"title": headline, "url": url}
            status, conf, img, body = presence_based_verdict(headline, art)
            try:
                record_activity(status, url)
            except:
                pass
            result.append(make_result(art, status, conf, img, body))

    return paginate_render(result, query, mode)

# ==================================================
# SUPPORT FUNCTIONS
# ==================================================
def get_title_from_url(url):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        t = soup.find("title")
        return normalize_text(t.get_text()) if t else url
    except:
        return url

def make_result(art, status, conf, img, body):
    return {
        "title": art.get("title"),
        "url": art.get("url"),
        "image": img or art.get("image") or "/static/images/news_placeholder.jpg",
        "source": base_domain(art.get("url") or ""),
        "status": status,
        "confidence": conf,
        "description": (body[:240] + "...") if body else ""
    }

def paginate_render(result, query, mode):
    page = int(request.args.get("page", 1))
    start = (page - 1) * RESULTS_PER_PAGE
    end = start + RESULTS_PER_PAGE
    paged = result[start:end]

    return render_template(
        "factchecks.html",
        result=paged,
        query=query,
        mode=mode,
        page=page,
        total=len(result),
        per_page=RESULTS_PER_PAGE
    )

# Provide now() to Jinja templates so templates can call `now()`
@app.context_processor
def inject_now():
    return {'now': datetime.now}

# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    app.run(debug=True)
