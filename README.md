# YouTube Playlist RAG

Chat with a YouTube playlist. Paste a playlist link once, then ask questions in plain language. Every answer comes with clickable timestamps, and the video plays right on the page at the exact moment the answer is drawn from.

I built this to stop scrubbing through hours of lectures looking for "that one part where they explained X."

## What it does

- Turns every video in a playlist into searchable text, with timestamps
- Answers your questions using only what was actually said in those videos
- Links each answer to the moment it came from, and plays it in an embedded player
- Remembers the playlist, so you only process it once
- Keeps your past chats in a History panel

## How it works

```
Playlist link
   -> yt-dlp downloads the audio
   -> ffmpeg slices it into 10-minute pieces
   -> Groq Whisper transcribes each piece, with timestamps
   -> the text is split into ~180-word chunks
   -> a local model (all-MiniLM-L6-v2) turns each chunk into a vector
   -> the vectors are stored in Qdrant

Your question
   -> turned into a vector
   -> Qdrant finds the 6 closest chunks from your playlist
   -> Llama 3.3 70B (on Groq) writes an answer from those chunks only
   -> you get the answer plus source links like youtu.be/<id>?t=<seconds>
```

Ingesting is the slow part and happens once. Chatting afterwards only searches what is already saved. Re-adding the same playlist skips any video that already has a transcript.

## What you need

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/) (both `ffmpeg` and `ffprobe`)
- [Deno](https://deno.com/), which yt-dlp now needs to download from YouTube reliably
- A free [Groq API key](https://console.groq.com/)
- Optional: a [Qdrant Cloud](https://cloud.qdrant.io/) cluster. Without one, the index is stored locally in the `data` folder.

Quick installs on Windows:

```
winget install ffmpeg
winget install DenoLand.Deno
```

On macOS use `brew install ffmpeg deno`. On Ubuntu use `sudo apt install ffmpeg`.

## Setup

```
git clone https://github.com/CYBER-CORE-DEV/Youtube-RAG-Playlist.git
cd Youtube-RAG-Playlist
uv sync
```

Copy `.env.example` to `.env` and fill it in:

```
GROQ_API_KEY=your_groq_key
QDRANT_URL=              # leave empty to store the index locally
QDRANT_API_KEY=          # only needed for Qdrant Cloud
```

Never commit your `.env` file. It is already in `.gitignore`.

## Run it

Start the backend:

```
uv run uvicorn backend:app --port 8001
```

In a second terminal, serve the page:

```
cd src/day16/frontend
python -m http.server 3000
```

Open http://127.0.0.1:3000.

On first use, paste a public playlist link and click **Add**. A long playlist can take a while, so keep the tab open until it says the playlist is ready. After that, the page opens straight into chat.

Open the page through that `http://` address rather than double-clicking `index.html`. YouTube blocks its embedded player on `file://` pages.

## Good to know

- **One playlist at a time.** Adding a different playlist replaces the current one. Old transcripts stay on disk, so switching back is quick.
- **Don't use `--reload` while ingesting.** The auto-restart can interrupt a long run.
- **Some videos can't be embedded.** Their owners disabled it. Use the "Open on YouTube" link under the player.
- **Transcripts are automatic.** Names and technical terms are sometimes misheard, so rephrase your question if an answer seems off.
- **Chat history lives in your browser**, not on the server.
- **The frontend expects the backend at `http://127.0.0.1:8001`.** If you change the port, edit the `API` line near the top of the script in `index.html`.

## Settings

All optional, set in `.env`:

| Variable | Default | What it does |
|---|---|---|
| `WHISPER_MODEL` | `whisper-large-v3-turbo` | Speech-to-text model on Groq |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model that writes the answers |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `QDRANT_COLLECTION` | `youtube_playlist` | Collection name in Qdrant |
| `FFMPEG_LOCATION` | not set | Folder containing ffmpeg and ffprobe, if they are not on your PATH |

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/status` | Is a playlist saved and ready? |
| `POST /api/ingest` | Download, transcribe and index a playlist. Safe to call again. |
| `POST /api/chat` | Ask a question. Returns an answer and its sources. |

Interactive docs are at http://127.0.0.1:8001/docs while the backend is running.

## Troubleshooting

**"ffmpeg not found" on startup.** Install it, or put `ffmpeg` and `ffprobe` next to `backend.py`, or set `FFMPEG_LOCATION`. Open a new terminal after installing so PATH updates.

**`HTTP Error 403` or "No supported JavaScript runtime" while downloading.** Install Deno, open a new terminal, and update yt-dlp with `uv add "yt-dlp[default]" --upgrade-package yt-dlp`. Then click Add again. Finished videos are skipped and only the failed ones retry.

**"No playlist saved yet."** Click **Add playlist** and ingest one first.

**Answers feel off or incomplete.** Check the video count under the top bar. If it is lower than your playlist, some videos failed. Add the same link again to retry them.

**Rate limit errors from Groq.** The backend waits and retries on its own. Long playlists on the free tier just take longer.

## Built with

FastAPI, yt-dlp, Groq (Whisper and Llama), sentence-transformers, Qdrant, and a single plain HTML page for the frontend.