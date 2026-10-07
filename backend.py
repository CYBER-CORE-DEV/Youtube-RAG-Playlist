"""
FastAPI Backend for YouTube Playlist RAG Chatbot
"""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchAny, PayloadSchemaType, PointStruct, VectorParams,
)
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="YouTube Playlist RAG Backend")

# Wildcard origins cannot be combined with credentials in browsers, and this
# API uses no cookies, so credentials are disabled.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Config & Initialization ----------
# Anchor data to this file, not the current working directory, so the same
# data folder is used no matter where the server is launched from.
BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data"
AUDIO_DIR = DATA / "audio"
TRANS_DIR = DATA / "transcripts"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANS_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "youtube_playlist")

AUDIO_SLICE_SEC = 600
CHUNK_WORDS = 180
OVERLAP_SEGMENTS = 2
TOP_K = 6

# ---------- ffmpeg discovery ----------
# Order: FFMPEG_LOCATION env var -> ffmpeg(.exe)/ffprobe(.exe) next to this
# file -> system PATH. The chosen folder is prepended to PATH so yt-dlp,
# subprocess calls and ffprobe all find it.
def _setup_ffmpeg():
    candidates = []
    if os.getenv("FFMPEG_LOCATION"):
        candidates.append(Path(os.environ["FFMPEG_LOCATION"]))
    candidates += [BASE_DIR, BASE_DIR / "bin"]
    for folder in candidates:
        if (folder / "ffmpeg.exe").exists() or (folder / "ffmpeg").exists():
            os.environ["PATH"] = str(folder) + os.pathsep + os.environ["PATH"]
            break
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} not found. Install ffmpeg, add its 'bin' folder to PATH, "
            f"set FFMPEG_LOCATION, or place ffmpeg and ffprobe next to {Path(__file__).name}."
        )


_setup_ffmpeg()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("Error: GROQ_API_KEY is missing.")

groq = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
print(f"Loading local embedding model ({EMBED_MODEL_NAME})...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Local Qdrant (path mode) is single-process and not safe for concurrent use,
# so all vector-store access goes through one shared client and one lock.
qdrant_lock = threading.RLock()
ingest_lock = threading.Lock()


# ---------- Core Logic Functions ----------
SKIP_TITLES = {"[Private video]", "[Deleted video]"}


def list_playlist(url: str) -> dict:
    """Read the playlist's id, title and video list WITHOUT downloading anything."""
    opts = {"extract_flat": "in_playlist", "ignoreerrors": True, "quiet": True, "noprogress": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        raise RuntimeError("Could not read the playlist/video. Check the URL and that it is public.")
    videos = [
        {"id": e["id"], "title": e.get("title") or e["id"]}
        for e in (info.get("entries") or [info])
        if e and e.get("id") and e.get("title") not in SKIP_TITLES
    ]
    if not videos:
        raise RuntimeError("No available videos found in that link.")
    return {"id": info.get("id") or videos[0]["id"],
            "title": info.get("title") or videos[0]["title"],
            "videos": videos}


def download_audio(video_ids: list[str]):
    """Download audio only for the given videos. We track what is done by checking
    the files on disk, so no download_archive is needed."""
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "64",
        }],
        "ignoreerrors": True,
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={i}" for i in video_ids])


# ---------- Saved playlist info (so the app remembers it between runs) ----------
META_FILE = DATA / "playlist.json"


def read_meta() -> dict:
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_meta(meta: dict):
    tmp = META_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    tmp.replace(META_FILE)


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _retry_after(err: Exception) -> float | None:
    try:
        return float(err.response.headers.get("retry-after"))
    except Exception:
        return None


def transcribe_slice(path: Path, retries: int = 5) -> list[dict]:
    last_err = None
    for attempt in range(retries):
        try:
            with open(path, "rb") as f:
                resp = groq.audio.transcriptions.create(
                    file=(path.name, f.read()),
                    model=WHISPER_MODEL,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            return [{"start": _get(s, "start"), "end": _get(s, "end"),
                     "text": _get(s, "text").strip()} for s in (resp.segments or [])]
        except (RateLimitError, APIConnectionError) as e:
            last_err = e
            wait = _retry_after(e) or 20 * (attempt + 1)
        except APIStatusError as e:
            if e.status_code < 500:
                # Bad key, bad request, file too large... retrying won't help.
                raise
            last_err = e
            wait = 20 * (attempt + 1)

        if attempt < retries - 1:
            print(f"Transcription retry {attempt + 1}/{retries - 1} for {path.name} in {wait:.0f}s ({last_err})")
            time.sleep(wait)
    raise RuntimeError(f"Failed to transcribe {path.name} after {retries} attempts: {last_err}")


def transcribe_audio(audio: Path) -> list[dict]:
    segments = []
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(audio), "-f", "segment",
             "-segment_time", str(AUDIO_SLICE_SEC), "-reset_timestamps", "1",
             "-c", "copy", str(Path(tmp) / "part_%03d.mp3")],
            check=True,
        )
        for i, part in enumerate(sorted(Path(tmp).glob("part_*.mp3"))):
            offset = i * AUDIO_SLICE_SEC
            for s in transcribe_slice(part):
                segments.append({"start": s["start"] + offset,
                                 "end": s["end"] + offset,
                                 "text": s["text"]})
    return segments


def transcribe(videos: list[dict]) -> list[dict]:
    """Transcribe each video; one failure no longer aborts the whole playlist.
    Returns a list of {id, title, error} for videos that failed."""
    failed = []
    for v in videos:
        out = TRANS_DIR / f"{v['id']}.json"
        audio = AUDIO_DIR / f"{v['id']}.mp3"
        if out.exists():
            continue
        if not audio.exists():
            failed.append({**v, "error": "audio file missing (download failed or was skipped)"})
            continue
        try:
            segs = transcribe_audio(audio)
        except Exception as e:
            print(f"Transcription failed for {v['id']}: {e}")
            failed.append({**v, "error": str(e)})
            continue
        # Write atomically so an interrupted run can't leave a corrupt JSON
        # file that would later be mistaken for a finished transcript.
        tmp_out = out.with_suffix(".json.tmp")
        tmp_out.write_text(json.dumps({**v, "segments": segs}, ensure_ascii=False), encoding="utf-8")
        tmp_out.replace(out)
    return failed


def chunk_segments(segs: list[dict]) -> list[dict]:
    chunks, cur = [], []
    for s in segs:
        cur.append(s)
        if sum(len(x["text"].split()) for x in cur) >= CHUNK_WORDS:
            chunks.append({"start": cur[0]["start"], "text": " ".join(x["text"] for x in cur)})
            cur = cur[-OVERLAP_SEGMENTS:]
    if cur and (not chunks or len(cur) > OVERLAP_SEGMENTS):
        chunks.append({"start": cur[0]["start"], "text": " ".join(x["text"] for x in cur)})
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    return embedder.encode(texts, show_progress_bar=False).tolist()


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """One shared client. Creating a new local client per call fails with
    'Storage folder is already accessed by another instance'."""
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(path=str(DATA / "qdrant"))


def ensure_collection(client: QdrantClient, dim: int):
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


_index_ready = False


def ensure_payload_index(client: QdrantClient):
    """Qdrant Cloud refuses to filter on a field that has no index.
    Creating an index that already exists is harmless. Local mode needs none."""
    global _index_ready
    if _index_ready or not QDRANT_URL:
        return
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="video_id",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )
    _index_ready = True


def point_id(video_id: str, i: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}:{i}"))


def index_videos(videos: list[dict]):
    """Index only the videos from this request (upserts are idempotent, but
    re-embedding every old transcript on each ingest is slow)."""
    client = get_client()
    for v in videos:
        f = TRANS_DIR / f"{v['id']}.json"
        if not f.exists():
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        chunks = chunk_segments(doc["segments"])
        if not chunks:
            continue
        vectors = embed([c["text"] for c in chunks])

        points = [
            PointStruct(
                id=point_id(doc["id"], i),
                vector=vec,
                payload={
                    "video_id": doc["id"],
                    "title": doc["title"],
                    "start": int(c["start"]),
                    "text": c["text"],
                },
            )
            for i, (c, vec) in enumerate(zip(chunks, vectors))
        ]
        with qdrant_lock:
            ensure_collection(client, len(vectors[0]))
            for j in range(0, len(points), 100):
                client.upsert(collection_name=COLLECTION, points=points[j:j + 100])


def reset_index():
    global _index_ready
    client = get_client()
    with qdrant_lock:
        _index_ready = False
        if client.collection_exists(COLLECTION):
            client.delete_collection(COLLECTION)


def fmt_time(sec: int) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------- API Schemas & Endpoints ----------
class IngestRequest(BaseModel):
    url: str = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    # list of (question, answer) pairs; malformed pairs now give a clean 422
    # instead of crashing with a ValueError while unpacking.
    history: list[tuple[str, str]] = []


SYSTEM_PROMPT = (
    "You are an assistant that answers questions about a YouTube playlist using ONLY "
    "the transcript excerpts provided in each message. Rules:\n"
    "- Ground every claim in the excerpts and cite like [Video title @ mm:ss].\n"
    "- If the excerpts do not contain the answer, say so plainly; do not guess.\n"
    "- When several excerpts are relevant, synthesize them into one clear answer.\n"
    "- Transcripts are auto-generated and may contain errors; use judgment on names/terms."
)


@app.get("/api/status")
def api_status():
    """Tells the frontend whether a playlist is already saved and ready to chat with."""
    meta = read_meta()
    ready = False
    if meta.get("indexed"):
        with qdrant_lock:
            ready = get_client().collection_exists(COLLECTION)
    return {"ready": ready, "title": meta.get("title"),
            "video_count": len(meta.get("videos", [])), "url": meta.get("url")}


@app.post("/api/ingest")
def api_ingest(req: IngestRequest):
    """Safe to call again and again: only videos without a saved transcript are
    downloaded/transcribed, and the index is only rebuilt when it must be."""
    if not ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An ingest is already running. Please wait for it to finish.")
    try:
        info = list_playlist(req.url)
        videos = info["videos"]
        has = lambda v: (TRANS_DIR / f"{v['id']}.json").exists()
        missing = [v for v in videos if not has(v)]

        failed = []
        if missing:
            need_audio = [v["id"] for v in missing if not (AUDIO_DIR / f"{v['id']}.mp3").exists()]
            if need_audio:
                download_audio(need_audio)
            failed = transcribe(missing)
        failed_ids = {f["id"] for f in failed}
        newly = [v for v in missing if v["id"] not in failed_ids]

        meta = read_meta()
        with qdrant_lock:
            has_collection = get_client().collection_exists(COLLECTION)
        rebuild = meta.get("playlist_id") != info["id"] or not meta.get("indexed") or not has_collection
        if rebuild:
            reset_index()  # the index only ever holds the current playlist
            index_videos([v for v in videos if has(v)])
        elif newly:
            index_videos(newly)

        ready_videos = [v for v in videos if has(v)]
        write_meta({"playlist_id": info["id"], "title": info["title"], "url": req.url,
                    "videos": ready_videos, "indexed": bool(ready_videos)})

        if not missing and not rebuild:
            message = f"Already saved. {len(ready_videos)} videos are ready, nothing was downloaded."
        else:
            message = f"Playlist ready: {len(ready_videos)} of {len(videos)} videos."
        resp = {"status": "success", "message": message}
        if failed:
            resp["failed"] = [{"id": f["id"], "title": f["title"], "error": f["error"]} for f in failed]
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ingest_lock.release()


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    client = get_client()
    try:
        # Only the saved playlist's videos may be used. Anything left in the index
        # from older playlists is ignored, even if it is still stored there.
        meta = read_meta()
        allowed = [v["id"] for v in meta.get("videos", [])]
        if not meta.get("indexed") or not allowed:
            raise HTTPException(status_code=400, detail="No playlist saved yet. Add your playlist first.")

        with qdrant_lock:
            if not client.collection_exists(COLLECTION):
                raise HTTPException(status_code=400, detail="Collection not found. Please ingest a playlist first.")

            # Only borrow the previous question for short follow-ups ("and why?"),
            # otherwise it drags retrieval toward the old topic.
            if req.history and len(req.message.split()) <= 6:
                search_q = f"{req.history[-1][0]} {req.message}"
            else:
                search_q = req.message

            ensure_payload_index(client)
            result = client.query_points(
                collection_name=COLLECTION,
                query=embed([search_q])[0],
                limit=TOP_K,
                with_payload=True,
                query_filter=Filter(must=[FieldCondition(key="video_id", match=MatchAny(any=allowed))]),
            )
        hits = [p.payload for p in result.points]

        context = "\n\n".join(
            f"[{h['title']} @ {fmt_time(h['start'])}]\n{h['text']}" for h in hits
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for pq, pa in req.history[-3:]:
            messages += [{"role": "user", "content": pq}, {"role": "assistant", "content": pa}]
        messages.append({"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {req.message}"})

        kwargs = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.2,
        }
        if REASONING_EFFORT and "llama-3" not in LLM_MODEL:
            kwargs["extra_body"] = {"reasoning_effort": REASONING_EFFORT}

        resp = groq.chat.completions.create(**kwargs)
        answer = resp.choices[0].message.content

        # Format sources uniquely
        sources = []
        seen = set()
        for h in hits:
            key = (h["video_id"], h["start"])
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "title": h["title"],
                "timestamp": fmt_time(h["start"]),
                "url": f"https://youtu.be/{h['video_id']}?t={h['start']}"
            })

        return {"answer": answer, "sources": sources}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))