#!/usr/bin/env python
"""Fetch Weibo group feed and save daily HTML + Markdown summary with images & comments."""

import html as html_mod
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

# Config
COOKIE_FILE = Path(__file__).resolve().parent / ".weibo_cookie"
OUTPUT_DIR = Path("C:/Users/PenPen/Desktop/weibo")
CHINA_TZ = timezone(timedelta(hours=8))
GROUP_ID = "110007448273554"

FEED_URL = "https://weibo.com/ajax/feed/friendstimeline"
LONGTEXT_URL = "https://weibo.com/ajax/statuses/longtext"
COMMENTS_URL = "https://weibo.com/ajax/statuses/buildComments"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"https://weibo.com/mygroups?gid={GROUP_ID}",
}

IMG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://weibo.com/",
}

RATE_LIMIT = 3
COMMENT_LIMIT = 10  # max top-level comments per post

URL_RE = re.compile(r'(https?://[^\s<>"]+)')
TOPIC_RE = re.compile(r'#([^#]+?)#')
MENTION_RE = re.compile(r'@(\S+)')


# ── cookie & session ──────────────────────────────────────────────

def load_cookie():
    if not COOKIE_FILE.exists():
        print(f"ERROR: Cookie file not found at {COOKIE_FILE}")
        print("Create this file with your Weibo cookie string from browser DevTools.")
        print("1. Go to https://weibo.com and log in")
        print("2. F12 -> Network -> refresh -> click any XHR request")
        print("3. Copy the 'Cookie' header value")
        print("4. Paste into .weibo_cookie file (one line)")
        sys.exit(1)
    cookie = COOKIE_FILE.read_text().strip()
    cookie = re.sub(r'SSOLoginState=\d+;?\s*', '', cookie)
    return cookie


def extract_xsrf(cookie_str):
    m = re.search(r'XSRF-TOKEN=([^;]+)', cookie_str)
    return m.group(1).strip() if m else ""


def create_session(cookie_str):
    session = requests.Session()
    session.headers.update(HEADERS)
    xsrf = extract_xsrf(cookie_str)
    if xsrf:
        session.headers["X-XSRF-TOKEN"] = xsrf
    for item in cookie_str.split(";"):
        item = item.strip()
        if "=" in item:
            key, _, value = item.partition("=")
            session.cookies.set(key.strip(), value.strip())
    return session


def api_get(session, url, params=None, retries=3):
    for attempt in range(retries):
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 432:
            wait = (2 ** attempt) * RATE_LIMIT
            print(f"  Rate limited (432), waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


# ── feed fetching ──────────────────────────────────────────────────

def fetch_feed(session, since_id=0, max_id=""):
    params = {
        "list_id": GROUP_ID, "refresh": 4,
        "since_id": since_id, "max_id": max_id,
        "count": 25, "fid": GROUP_ID,
    }
    resp = api_get(session, FEED_URL, params=params)
    data = resp.json()
    return data.get("statuses", []), data.get("since_id", ""), data.get("max_id", "")


# ── long text expansion ────────────────────────────────────────────

def fetch_long_text(session, post_id):
    resp = api_get(session, LONGTEXT_URL, params={"id": post_id})
    data = resp.json()
    if data.get("ok") == 1:
        return data.get("data", {}).get("longTextContent", "")
    return ""


# ── comment fetching ───────────────────────────────────────────────

def fetch_comments(session, post_id, post_uid, count=COMMENT_LIMIT):
    """Fetch top comments for a post. Returns list of comment dicts."""
    params = {
        "id": post_id, "is_reload": 1, "fetch_level": 0,
        "count": count, "max_id": 0,
        "is_show_bulletin": 2, "is_mix": 1,
        "uid": post_uid, "locale": "zh-CN",
    }
    try:
        resp = api_get(session, COMMENTS_URL, params=params, retries=2)
        data = resp.json()
        if data.get("ok") != 1:
            return []
        comments_list = data.get("data", [])
        if isinstance(comments_list, dict):
            comments_list = comments_list.get("data", [])
        if not isinstance(comments_list, list):
            return []
        comments = []
        for c in comments_list:
            # Some comments use a {type, data: {...}} wrapper structure
            inner = c.get("data", c) if isinstance(c.get("data"), dict) else c
            text = inner.get("text_raw", "") or inner.get("text", "")
            if not text:
                continue  # skip empty/deleted comments
            comments.append({
                "author": inner.get("user", {}).get("screen_name", "Unknown"),
                "text": text,
                "likes": inner.get("like_count", 0) or inner.get("like_counts", 0),
                "created_at": inner.get("created_at", ""),
            })
        return comments
    except Exception as e:
        print(f"    Comment fetch failed for {post_id}: {e}")
        return []


# ── image downloading ──────────────────────────────────────────────

def _download_one(url, dest_path):
    """Download a single image to disk. Returns (url, local_filename) or (url, None)."""
    try:
        resp = requests.get(url, headers=IMG_HEADERS, timeout=15)
        if resp.status_code == 200:
            dest_path.write_bytes(resp.content)
            return url, dest_path.name
        return url, None
    except Exception:
        return url, None


def download_images(posts, images_dir):
    """Download all images from all posts (main + retweeted) to images_dir using threads."""
    # Collect all unique image URLs
    url_map = {}  # url -> local filename
    for post in posts:
        for url in post.get("pics", []):
            if url not in url_map:
                ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
                if "?" in ext:
                    ext = ".jpg"
                fname = f"{len(url_map):04d}{ext}"
                url_map[url] = fname
        if post.get("video") and post["video"].get("poster"):
            url = post["video"]["poster"]
            if url and url not in url_map:
                ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
                if "?" in ext:
                    ext = ".jpg"
                fname = f"{len(url_map):04d}{ext}"
                url_map[url] = fname
        if post.get("retweeted"):
            for url in post["retweeted"].get("pics", []):
                if url not in url_map:
                    ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
                    if "?" in ext:
                        ext = ".jpg"
                    fname = f"{len(url_map):04d}{ext}"
                    url_map[url] = fname

    if not url_map:
        print("  No images to download.")
        return {}

    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {len(url_map)} images ({images_dir})...")

    failed = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_download_one, url, images_dir / fname): url
                   for url, fname in url_map.items()}
        for future in as_completed(futures):
            url, result = future.result()
            if result is None:
                failed += 1
            # keep mapping regardless of success/failure for consistent path references

    if failed:
        print(f"  {failed} image(s) failed to download (will use remote URL fallback).")

    # Build lookup table: remote URL -> local relative path
    lookup = {}
    for url, fname in url_map.items():
        local_file = images_dir / fname
        if local_file.exists():
            lookup[url] = f"{images_dir.name}/{fname}"
        else:
            lookup[url] = url  # fallback to remote
    return lookup


def relink_images(post, img_lookup):
    """Replace remote image URLs with local paths using the lookup table."""
    post["pics"] = [img_lookup.get(u, u) for u in post.get("pics", [])]
    if post.get("video") and post["video"].get("poster"):
        post["video"]["poster"] = img_lookup.get(post["video"]["poster"], post["video"]["poster"])
    if post.get("retweeted"):
        relink_images(post["retweeted"], img_lookup)


# ── post parsing ───────────────────────────────────────────────────

def parse_post(post):
    user = post.get("user", {})

    # Images via pic_ids + pic_infos
    pics = []
    pic_infos = post.get("pic_infos", {})
    for pid in post.get("pic_ids", []):
        info = pic_infos.get(pid, {})
        for size in ("large", "largest", "bmiddle", "thumbnail", "original"):
            size_info = info.get(size, {})
            url = size_info.get("url", "") if isinstance(size_info, dict) else ""
            if url:
                pics.append(url.replace("http://", "https://"))
                break

    # mix_media_info images
    for item in post.get("mix_media_info", {}).get("items", []):
        if item.get("type") == "pic":
            data = item.get("data", {})
            for size in ("large", "bmiddle", "thumbnail"):
                size_info = data.get(size, {})
                url = size_info.get("url", "") if isinstance(size_info, dict) else ""
                if url:
                    pics.append(url.replace("http://", "https://"))
                    break

    # Video
    page_info = post.get("page_info", {})
    video = None
    if page_info.get("object_type") == "video":
        poster = page_info.get("page_pic", "")
        if isinstance(poster, dict):
            poster = poster.get("url", "")
        media = page_info.get("media_info", {})
        mp4_url = (
            media.get("mp4_720p_mp4", "") or media.get("mp4_hd_url", "")
            or media.get("mp4_sd_url", "") or media.get("stream_url_hd", "")
            or media.get("stream_url", "")
        )
        video = {
            "poster": poster.replace("http://", "https://") if poster else "",
            "mp4": mp4_url.replace("http://", "https://") if mp4_url else "",
            "page_url": page_info.get("page_url", ""),
        }

    is_long = post.get("isLongText", False)
    text_raw = post.get("text_raw", "") or ""
    text_html = post.get("text", "")

    retweeted = None
    if post.get("retweeted_status"):
        retweeted = parse_post(post["retweeted_status"])

    return {
        "author": user.get("screen_name", "Unknown"),
        "author_uid": user.get("id", ""),
        "avatar": user.get("avatar_large", "").replace("http://", "https://"),
        "text_raw": text_raw,
        "text_html": text_html,
        "is_long": is_long,
        "created_at": post.get("created_at", ""),
        "reposts": post.get("reposts_count", 0),
        "comments_count": post.get("comments_count", 0),
        "likes": post.get("attitudes_count", 0),
        "pics": pics,
        "video": video,
        "retweeted": retweeted,
        "bid": post.get("bid", ""),
        "source": post.get("source", ""),
        "id": post.get("id", 0),
        "mid": post.get("mid", "") or post.get("idstr", ""),
        "weibo_url": (
            f"https://weibo.com/{user.get('id', '')}/{post.get('bid') or post.get('mid', '')}"
            if (post.get("bid") or post.get("mid")) else ""
        ),
        "comments": [],  # populated later
    }


def is_within_24h(created_at_str, cutoff):
    if not created_at_str:
        return True
    if any(x in created_at_str for x in ["刚刚", "秒前", "分钟前", "小时前"]):
        return True
    try:
        return datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y") > cutoff
    except (ValueError, TypeError):
        pass
    try:
        return datetime.fromisoformat(created_at_str.replace("Z", "+00:00")) > cutoff
    except (ValueError, TypeError):
        pass
    try:
        return datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=CHINA_TZ) > cutoff
    except (ValueError, TypeError):
        pass
    return "昨天" in created_at_str or True


# ── HTML output ────────────────────────────────────────────────────

def format_text_html(text, is_long=False):
    if is_long and text:
        text = re.sub(r'<span[^>]*class="surl-text"[^>]*>.*?</span>', '', text)
        text = re.sub(r'<img[^>]*class="emoji"[^>]*/?>', '', text)
        text = re.sub(r'<span[^>]*>', '', text)
        text = re.sub(r'</span>', '', text)
        text = re.sub(r'\s+style="[^"]*"', '', text)
        text = re.sub(r'class="[^"]*"', '', text)
        text = re.sub(r'href="//', 'href="https://', text)
        text = re.sub(r'<a ', r'<a target="_blank" rel="noopener" ', text)
        return text.replace('\n', '')

    text = html_mod.escape(text)
    text = URL_RE.sub(r'<a href="\1" target="_blank" rel="noopener">\1</a>', text)
    text = TOPIC_RE.sub(
        r'<a href="https://weibo.com/search?q=%23\1%23" target="_blank" rel="noopener">#\1#</a>', text)
    text = MENTION_RE.sub(
        r'<a href="https://weibo.com/n/\1" target="_blank" rel="noopener">@\1</a>', text)
    text = re.sub(r'\[([a-zA-Z一-鿿]+)\]', r'<span class="emoji">[\1]</span>', text)
    return text.replace("\n", "<br>")


def _pics_html(pics):
    if not pics:
        return ""
    parts = ['<div class="media-grid">']
    for url in pics:
        parts.append(
            f'<a href="{url}" target="_blank">'
            f'<img src="{url}" loading="lazy" alt="" onerror="this.style.display=\'none\'">'
            f'</a>')
    parts.append('</div>')
    return "\n".join(parts)


def _video_html(video):
    if not video:
        return ""
    parts = ['<div class="video-card">']
    if video["poster"]:
        parts.append(
            f'<a href="{video["page_url"] or video["mp4"] or "#"}" target="_blank" rel="noopener">'
            f'<img src="{video["poster"]}" class="video-poster" loading="lazy" alt="">'
            f'<span class="play-icon">▶</span></a>')
    parts.append(
        f'<div class="video-link">🎬 <a href="{video["mp4"] or video["page_url"] or "#"}" '
        f'target="_blank" rel="noopener">Watch video</a></div>')
    parts.append('</div>')
    return "\n".join(parts)


def _comments_html(comments):
    if not comments:
        return ""
    parts = ['<div class="comments">']
    for c in comments:
        parts.append('<div class="comment">')
        parts.append(f'<span class="comment-author">{html_mod.escape(c["author"])}</span>')
        parts.append(f'<span class="comment-text">{format_text_html(c["text"])}</span>')
        if c["likes"]:
            parts.append(f'<span class="comment-likes">❤️ {c["likes"]}</span>')
        parts.append('</div>')
    parts.append('</div>')
    return "\n".join(parts)


def _retweet_html(retweet):
    parts = ['<div class="retweet">']
    parts.append(
        f'<div class="retweet-author">🔁 <a href="https://weibo.com/u/{retweet["author_uid"]}" '
        f'target="_blank" rel="noopener">@{retweet["author"]}</a></div>')
    display = retweet["text_html"] if retweet["is_long"] else retweet["text_raw"]
    parts.append(f'<div class="retweet-text">{format_text_html(display, retweet["is_long"])}</div>')
    parts.append(_pics_html(retweet["pics"]))
    parts.append(_video_html(retweet.get("video")))
    parts.append(_comments_html(retweet.get("comments", [])))
    parts.append('</div>')
    return "\n".join(parts)


def format_post_html(post):
    parts = ['<article class="post">']
    # Header
    parts.append('<div class="post-header">')
    if post["avatar"]:
        parts.append(f'<img class="avatar" src="{post["avatar"]}" alt="" width="40" height="40">')
    parts.append('<div class="post-meta">')
    parts.append(
        f'<span class="author"><a href="https://weibo.com/u/{post["author_uid"]}" '
        f'target="_blank" rel="noopener">{html_mod.escape(post["author"])}</a></span>')
    parts.append(f'<span class="time">{html_mod.escape(post["created_at"])}</span>')
    if post["source"]:
        parts.append(f'<span class="source">via {html_mod.escape(post["source"])}</span>')
    parts.append('</div></div>')
    # Text
    display = post["text_html"] if post["is_long"] else post["text_raw"]
    parts.append(f'<div class="post-text">{format_text_html(display, post["is_long"])}</div>')
    # Media
    parts.append(_pics_html(post["pics"]))
    parts.append(_video_html(post.get("video")))
    # Retweet
    if post.get("retweeted"):
        parts.append(_retweet_html(post["retweeted"]))
    # Comments
    parts.append(_comments_html(post.get("comments", [])))
    # Stats
    parts.append('<div class="post-stats">')
    stats = []
    if post["reposts"]:
        stats.append(f'<span class="stat">🔄 {post["reposts"]}</span>')
    if post["comments_count"]:
        stats.append(f'<span class="stat">💬 {post["comments_count"]}</span>')
    if post["likes"]:
        stats.append(f'<span class="stat">❤️ {post["likes"]}</span>')
    parts.append(" ".join(stats) if stats else "")
    if post["weibo_url"]:
        parts.append(f' <a class="permalink" href="{post["weibo_url"]}" target="_blank" rel="noopener">Open in Weibo →</a>')
    parts.append('</div>')
    parts.append('</article>')
    return "\n".join(parts)


CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;max-width:680px;margin:0 auto;padding:20px;background:#f5f5f5;color:#222}
h1{font-size:1.4em;margin-bottom:4px}
.summary-meta{color:#666;font-size:.9em;margin-bottom:24px}
.post{background:#fff;border-radius:10px;padding:18px 20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.post-header{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.avatar{border-radius:50%;width:40px;height:40px;object-fit:cover;flex-shrink:0}
.post-meta{display:flex;flex-direction:column;gap:2px}
.author{font-weight:600;font-size:.95em}
.author a{color:#222;text-decoration:none}
.author a:hover{text-decoration:underline}
.time{color:#999;font-size:.8em}
.source{color:#bbb;font-size:.75em}
.post-text{font-size:.95em;line-height:1.65;margin-bottom:12px;word-break:break-word}
.post-text a{color:#e65036;text-decoration:none}
.post-text a:hover{text-decoration:underline}
.emoji{color:#e65036;font-size:.9em}
.media-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:6px;margin-bottom:12px}
.media-grid a{display:block}
.media-grid img{width:100%;height:auto;max-height:400px;object-fit:cover;border-radius:6px;border:1px solid #eee}
.media-grid:has(a:only-child){grid-template-columns:1fr}
.media-grid:has(a:only-child) img{max-height:500px;object-fit:contain}
.video-card{position:relative;margin-bottom:12px;border-radius:6px;overflow:hidden;background:#000;display:inline-block;max-width:100%}
.video-card a{display:block;position:relative}
.video-poster{max-width:100%;max-height:400px;display:block;opacity:.8}
.play-icon{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:2.5em;color:#fff;text-shadow:0 0 8px rgba(0,0,0,.5);pointer-events:none}
.video-link{padding:6px 8px;font-size:.85em}
.video-link a{color:#4a9eff;text-decoration:none}
.video-link a:hover{text-decoration:underline}
.retweet{border:1px solid #e8e8e8;border-radius:8px;padding:14px 16px;margin-bottom:12px;background:#fafafa}
.retweet-author{font-weight:600;font-size:.85em;margin-bottom:6px}
.retweet-author a{color:#e65036;text-decoration:none}
.retweet-text{font-size:.9em;line-height:1.5;color:#555}
.comments{border-top:1px solid #f0f0f0;margin-top:10px;padding-top:8px}
.comment{padding:4px 0;font-size:.85em;line-height:1.4}
.comment-author{font-weight:600;margin-right:6px;color:#555}
.comment-text{color:#444}
.comment-likes{color:#ccc;font-size:.8em;margin-left:6px}
.post-stats{font-size:.85em;color:#999;padding-top:8px;border-top:1px solid #f0f0f0;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.stat{white-space:nowrap}
.permalink{color:#e65036;text-decoration:none;margin-left:auto}
.permalink:hover{text-decoration:underline}
@media(max-width:500px){body{padding:10px}.post{padding:12px 14px}.media-grid{grid-template-columns:1fr}}
"""


def format_html(posts, date_str):
    parts = [
        "<!DOCTYPE html>", '<html lang="zh-CN">', "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="referrer" content="no-referrer">',
        f"<title>Weibo Summary — {date_str}</title>",
        f"<style>{CSS}</style>", "</head>", "<body>",
        f"<h1>Weibo Daily Summary — {date_str}</h1>",
        f'<p class="summary-meta"><strong>{len(posts)} posts</strong> from your timeline in the last 24 hours.</p>',
    ]
    for post in posts:
        parts.append(format_post_html(post))
    parts.append("</body></html>")
    return "\n".join(parts)


# ── Markdown output ────────────────────────────────────────────────

def _md_text(text):
    """Clean text for markdown: strip HTML, unescape, keep plain text."""
    text = re.sub(r'<[^>]+>', '', text)
    text = html_mod.unescape(text)
    return text.strip()


def _md_comments(comments):
    if not comments:
        return ""
    lines = ["", "**Comments:**"]
    for c in comments:
        likes = f" (❤️ {c['likes']})" if c["likes"] else ""
        lines.append(f"- **{c['author']}**{likes}: {_md_text(c['text'])}")
    return "\n".join(lines)


def _md_post(post, index):
    lines = [f"## {index}. {post['author']}", "", _md_text(post["text_raw"] or post["text_html"])]

    if post["pics"]:
        lines.append("")
        lines.append(f"📷 {len(post['pics'])} image(s):")
        for url in post["pics"]:
            lines.append(f"  ![]({url})")

    if post.get("video"):
        v = post["video"]
        lines.append("")
        lines.append(f"🎬 Video: {v['mp4'] or v['page_url']}")

    if post.get("retweeted"):
        rt = post["retweeted"]
        lines.append("")
        lines.append(f"> 🔁 **@{rt['author']}**: {_md_text(rt['text_raw'] or rt['text_html'])}")
        if rt.get("pics"):
            for url in rt["pics"]:
                lines.append(f"> ![]({url})")

    lines.append(_md_comments(post.get("comments", [])))

    detail = [f"- {post['created_at']}"]
    if post["reposts"]:
        detail.append(f"🔄 {post['reposts']}")
    if post["comments_count"]:
        detail.append(f"💬 {post['comments_count']}")
    if post["likes"]:
        detail.append(f"❤️ {post['likes']}")
    lines.append("")
    lines.append("  ".join(detail))
    if post["weibo_url"]:
        lines.append(f"- 🔗 {post['weibo_url']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def format_markdown(posts, date_str):
    lines = [
        f"# Weibo Daily Summary — {date_str}",
        "",
        f"**{len(posts)} posts** from your timeline in the last 24 hours.",
        "",
        "---",
        "",
    ]
    for i, post in enumerate(posts, 1):
        lines.append(_md_post(post, i))
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────

def main():
    print(f"[{datetime.now()}] Starting Weibo group summary (gid={GROUP_ID})...")

    cookie_str = load_cookie()
    session = create_session(cookie_str)

    print("  Checking authentication...")
    try:
        test_resp = session.get(f"https://weibo.com/mygroups?gid={GROUP_ID}", timeout=15)
        if test_resp.status_code == 302 or "passport" in test_resp.url.lower():
            print("ERROR: Cookie appears to be expired.")
            sys.exit(1)
        print(f"  Auth check OK (status {test_resp.status_code})")
    except Exception as e:
        print(f"  Pre-check warning (continuing): {e}")

    now = datetime.now(CHINA_TZ)
    cutoff = now - timedelta(hours=24)

    # ── Step 1: Fetch feed ──
    all_posts = []
    since_id, max_id = 0, ""
    for page in range(1, 31):
        print(f"  Fetching batch {page} (since_id={since_id}, max_id={max_id})...")
        try:
            batch, new_sid, new_mid = fetch_feed(session, since_id=since_id, max_id=max_id)
        except Exception as e:
            print(f"  API error: {e}")
            break
        if not batch:
            print("  No more posts returned.")
            break

        batch_in = 0
        for post in batch:
            parsed = parse_post(post)
            if is_within_24h(parsed["created_at"], cutoff):
                batch_in += 1
                all_posts.append(parsed)

        print(f"  Batch {page}: {len(batch)} raw, {batch_in} within 24h, {len(all_posts)} total")
        if batch_in / max(len(batch), 1) < 0.5:
            print("  Less than half of batch within 24h — stopping.")
            break
        if page == 1:
            since_id = ""
        if new_mid:
            max_id = new_mid
        else:
            print("  No max_id — stopping.")
            break
        time.sleep(RATE_LIMIT)

    # Deduplicate
    seen = set()
    unique = []
    for p in all_posts:
        key = (p["author"], p["text_raw"][:80])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    all_posts = unique

    # ── Step 2: Expand long text ──
    expand_list = []
    for p in all_posts:
        if p["is_long"] and p["id"]:
            expand_list.append(("main", p))
        if p.get("retweeted") and p["retweeted"]["is_long"] and p["retweeted"]["id"]:
            expand_list.append(("retweeted", p))

    if expand_list:
        print(f"  Fetching full text for {len(expand_list)} long posts...")
        for kind, p in expand_list:
            target = p if kind == "main" else p["retweeted"]
            try:
                full = fetch_long_text(session, target["id"])
                if full:
                    target["text_html"] = full
            except Exception as e:
                print(f"    Failed: {target['author']} — {e}")
            time.sleep(RATE_LIMIT * 0.5)

    # ── Step 3: Download images ──
    date_str = now.strftime("%Y-%m-%d")
    images_dir = OUTPUT_DIR / f"images_{date_str}"
    img_lookup = download_images(all_posts, images_dir)
    if img_lookup:
        for p in all_posts:
            relink_images(p, img_lookup)

    # ── Step 4: Fetch comments ──
    posts_for_comments = [p for p in all_posts if p["comments_count"] > 0]
    if posts_for_comments:
        print(f"  Fetching comments for {len(posts_for_comments)} posts...")
        fetched = 0
        for p in posts_for_comments:
            try:
                p["comments"] = fetch_comments(session, p["id"], p["author_uid"])
                if p["comments"]:
                    fetched += 1
            except Exception as e:
                pass
            time.sleep(RATE_LIMIT * 0.5)
        print(f"    Got comments for {fetched}/{len(posts_for_comments)} posts")

    # ── Step 5: Write output ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    html_file = OUTPUT_DIR / f"weibo_{date_str}.html"
    html_file.write_text(format_html(all_posts, date_str), encoding="utf-8")
    print(f"  HTML: {len(all_posts)} posts → {html_file}")

    md_file = OUTPUT_DIR / f"weibo_{date_str}.md"
    md_file.write_text(format_markdown(all_posts, date_str), encoding="utf-8")
    print(f"  Markdown: {len(all_posts)} posts → {md_file}")

    print(f"  Done!")


if __name__ == "__main__":
    main()
