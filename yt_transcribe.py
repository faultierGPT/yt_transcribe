#!/usr/bin/env python3
"""
yt_transcribe.py — download audio from a YouTube URL with yt-dlp and transcribe it
via the OpenAI Whisper API (model `whisper-1`). No local GPU/CPU inference.

Usage:
    # simplest: download + transcribe with the default API model (whisper-1)
    python yt_transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    # pick audio format, force a language, write to ./out
    python yt_transcribe.py URL --audio-format mp3 --language en --output-dir ./out

    # only download, skip transcription (handy for batch collection)
    python yt_transcribe.py URL --skip-transcribe

    # transcribe an audio file you already have
    python yt_transcribe.py --local-file podcast.mp3 --language en

Requirements:
    export OPENAI_API_KEY="sk-..."    # https://platform.openai.com/api-keys

Defaults:
    --model         whisper-1   (currently the only Whisper model exposed via the API)
    --language      auto-detect
    --output-dir    ./transcripts
    --audio-format  mp3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Defaults that match the rest of the project conventions.
DEFAULT_MODEL = "whisper-1"
DEFAULT_OUTPUT_DIR = Path("./transcripts")
DEFAULT_AUDIO_FORMAT = "mp3"

# OpenAI Whisper API hard limit: https://platform.openai.com/docs/api-reference/audio/createTranscription
OPENAI_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}


# --------------------------------------------------------------------------- #
# yt-dlp integration
# --------------------------------------------------------------------------- #
def yt_dlp_available() -> bool:
    return shutil.which("yt-dlp") is not None


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def download_audio(url: str, output_dir: Path, audio_format: str) -> Path:
    """
    Download audio only from `url` into `output_dir/<video_id>.<audio_format>`.

    Uses yt-dlp's external downloader chain (yt-dlp + ffmpeg) so we get a
    single, clean audio file even when the source is a video.

    Returns the absolute path to the downloaded audio file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # %(id)s.%(ext)s keeps filenames short and stable per video.
    output_template = str(output_dir / "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",                              # extract audio
        "--audio-format", audio_format,    # postprocess to mp3/m4a/...
        "--audio-quality", "0",            # best
        "--no-playlist",                   # default: single video, not whole playlist
        "-o", output_template,
        "--newline",                       # one progress line per event, easier to parse
        "--no-progress",                   # quiet stderr except errors
        url,
    ]
    print(f"[yt-dlp] downloading {url}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed for {url}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # yt-dlp writes  a file per video. Pick the most recently modified one.
    candidates = sorted(
        output_dir.glob("*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"yt-dlp reported success but no file appeared in {output_dir}")
    return candidates[0].resolve()


# --------------------------------------------------------------------------- #
# OpenAI Whisper API integration
# --------------------------------------------------------------------------- #
def _require_openai_client():
    """
    Lazily import the `openai` SDK so `--help` and `--skip-transcribe` still run
    instantly without pulling in the dependency.
    """
    try:
        import openai  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The `openai` Python package is not installed. "
            "Install it into the project venv:\n"
            "  python3 -m venv .venv && .venv/bin/pip install openai"
        ) from exc
    return openai


def _require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running:\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "Create one at https://platform.openai.com/api-keys"
        )
    return key


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: str | None,
) -> tuple[list[TranscriptSegment], dict]:
    """
    Transcribe `audio_path` via the OpenAI Whisper API.

    Returns (segments, info) where info carries detected language and duration
    in the same shape the faster-whisper path used to return, so downstream
    code (output writers) does not have to know which backend ran.
    """
    # 1. File-size guard — the API rejects anything > 25 MB.
    size = audio_path.stat().st_size
    if size > OPENAI_MAX_UPLOAD_BYTES:
        mb = size / 1024 / 1024
        raise RuntimeError(
            f"{audio_path.name} is {mb:.1f} MB, which exceeds the OpenAI Whisper API's "
            f"25 MB upload limit. Re-download with a lower bitrate, e.g.:\n"
            f"  yt-dlp -x --audio-format mp3 --audio-quality 9 -o '%(id)s.%(ext)s' <url>\n"
            f"or split the file locally with ffmpeg before retrying."
        )

    openai = _require_openai_client()
    api_key = _require_api_key()

    # 2. Build the client. Constructing inside the function lets `--help` and
    #    `--skip-transcribe` run without an OPENAI_API_KEY at all.
    client = openai.OpenAI(api_key=api_key)

    print(
        f"[whisper-api] uploading {audio_path.name} ({size/1024/1024:.1f} MB) "
        f"model={model_name!r}",
        file=sys.stderr,
    )

    # 3. Call the API. verbose_json + segment granularities gives us per-segment
    #    timestamps identical in shape to faster-whisper's output.
    with audio_path.open("rb") as fh:
        response = client.audio.transcriptions.create(
            model=model_name,
            file=(audio_path.name, fh),
            language=language,                              # None => auto-detect
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    # `response` is an openai.types.audio.TranscriptionVerbose; access as model
    # because pydantic models don't behave like dicts in older stubs.
    raw_segments = getattr(response, "segments", None) or []
    segments = [
        TranscriptSegment(start=float(s.start), end=float(s.end), text=s.text.strip())
        for s in raw_segments
    ]

    meta = {
        "language": getattr(response, "language", None),
        "language_probability": None,  # API does not expose this; field kept for shape compat
        "duration": float(getattr(response, "duration", 0.0) or 0.0),
    }

    print(
        f"[whisper-api] done: language={meta['language']} "
        f"duration={meta['duration']:.1f}s segments={len(segments)}",
        file=sys.stderr,
    )
    return segments, meta


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def write_outputs(
    audio_path: Path,
    segments: list[TranscriptSegment],
    meta: dict,
    output_dir: Path,
    source_url: str | None,
) -> tuple[Path, Path]:
    """Write transcript as both plain text and JSON with timestamps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    txt_path = output_dir / f"{stem}.txt"
    json_path = output_dir / f"{stem}.json"

    # Plain text: one segment per line, no timestamps. Easy to grep / paste.
    txt_path.write_text(
        "\n".join(seg.text for seg in segments) + "\n",
        encoding="utf-8",
    )

    # JSON: everything — segments, metadata, source URL.
    payload = {
        "source_url": source_url,
        "audio_file": str(audio_path.name),
        "meta": meta,
        "segments": [seg.to_dict() for seg in segments],
        "full_text": "\n".join(seg.text for seg in segments),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return txt_path, json_path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def process_url(
    url: str,
    output_dir: Path,
    audio_format: str,
    model_name: str,
    language: str | None,
    skip_transcribe: bool,
    keep_audio: bool,
) -> None:
    audio_path = download_audio(url, output_dir, audio_format)
    print(f"[ok] audio: {audio_path}", file=sys.stderr)

    if skip_transcribe:
        return

    segments, meta = transcribe_audio(audio_path, model_name, language)
    txt_path, json_path = write_outputs(audio_path, segments, meta, output_dir, url)
    print(f"[ok] transcript (txt): {txt_path}", file=sys.stderr)
    print(f"[ok] transcript (json): {json_path}", file=sys.stderr)

    if not keep_audio:
        audio_path.unlink(missing_ok=True)
        print(f"[cleanup] removed {audio_path}", file=sys.stderr)


def process_local_file(
    audio_path: Path,
    output_dir: Path,
    model_name: str,
    language: str | None,
) -> None:
    segments, meta = transcribe_audio(audio_path, model_name, language)
    txt_path, json_path = write_outputs(audio_path, segments, meta, output_dir, source_url=None)
    print(f"[ok] transcript (txt): {txt_path}", file=sys.stderr)
    print(f"[ok] transcript (json): {json_path}", file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download a YouTube URL's audio (yt-dlp) and transcribe it via "
            "the OpenAI Whisper API. Requires OPENAI_API_KEY in the environment."
        ),
    )
    p.add_argument(
        "urls",
        nargs="*",
        help="One or more YouTube URLs. Omit when --local-file is used.",
    )
    p.add_argument(
        "--local-file",
        type=Path,
        help="Transcribe a local audio file instead of downloading.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to put audio + transcripts (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "OpenAI Whisper model name. As of 2026 the public API exposes only "
            "`whisper-1`. Kept as a flag so a future model slug just works. "
            "Default: %(default)s"
        ),
    )
    p.add_argument(
        "--language",
        default=None,
        help="Force a language code (e.g. 'en', 'de'). Default: auto-detect.",
    )
    p.add_argument(
        "--audio-format",
        default=DEFAULT_AUDIO_FORMAT,
        help=(
            "yt-dlp audio format (mp3, m4a, wav, opus, ...). Use mp3 + low "
            "bitrate to stay under the 25 MB API limit. Default: %(default)s"
        ),
    )
    p.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Only download audio, do not transcribe.",
    )
    p.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the downloaded audio file after transcription (default: delete it).",
    )
    return p.parse_args(argv)


def require_tool(name: str, hint: str) -> None:
    if shutil.which(name) is None:
        print(f"[error] {name!r} not found on PATH. {hint}", file=sys.stderr)
        sys.exit(2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.urls and not args.local_file:
        print("[error] provide at least one URL or --local-file.", file=sys.stderr)
        parse_args(["--help"])
        return 2

    if args.urls and args.local_file:
        print("[error] pass URLs or --local-file, not both.", file=sys.stderr)
        return 2

    output_dir = args.output_dir.expanduser().resolve()

    if args.local_file:
        process_local_file(
            audio_path=args.local_file.expanduser().resolve(),
            output_dir=output_dir,
            model_name=args.model,
            language=args.language,
        )
        return 0

    require_tool("yt-dlp", "Install with: pipx install yt-dlp  (or: uv tool install yt-dlp)")
    require_tool("ffmpeg", "Install with: sudo apt install ffmpeg")

    for url in args.urls:
        try:
            process_url(
                url=url,
                output_dir=output_dir,
                audio_format=args.audio_format,
                model_name=args.model,
                language=args.language,
                skip_transcribe=args.skip_transcribe,
                keep_audio=args.keep_audio,
            )
        except Exception as exc:  # noqa: BLE001 - we want to keep going on multi-URL runs
            print(f"[error] {url}: {exc}", file=sys.stderr)
            continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
