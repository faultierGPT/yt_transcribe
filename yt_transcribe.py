#!/usr/bin/env python3
"""
yt_transcribe.py — download audio from a YouTube URL with yt-dlp and transcribe it.

Usage:
    python yt_transcribe.py URL [URL ...] [options]

Examples:
    # simplest: download + transcribe with default model (small)
    python yt_transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    # pick a larger model, force English, write everything into ./out
    python yt_transcribe.py URL --model medium --language en --output-dir ./out

    # only download, skip transcription (handy for batch collection)
    python yt_transcribe.py URL --skip-transcribe

    # transcribe an audio file you already have
    python yt_transcribe.py --local-file podcast.mp3 --model medium

Defaults:
    --model       tiny   (fastest, lowest quality; switch to small/medium/large-v3 for real use)
    --language    auto-detect
    --output-dir  ./transcripts
    --audio-format mp3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Defaults that match the rest of the project conventions.
DEFAULT_MODEL = "tiny"
DEFAULT_OUTPUT_DIR = Path("./transcripts")
DEFAULT_AUDIO_FORMAT = "mp3"


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

    # yt-dlp writes <filename>.<ext>; we know the ext is our audio_format unless
    # the source had no transcode-able audio stream (rare). Find the freshest file.
    candidates = sorted(
        output_dir.glob("*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"yt-dlp reported success but no file appeared in {output_dir}")
    return candidates[0].resolve()


# --------------------------------------------------------------------------- #
# faster-whisper integration
# --------------------------------------------------------------------------- #
def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: str | None,
) -> tuple[list[TranscriptSegment], dict]:
    """
    Transcribe `audio_path` with faster-whisper.

    Returns (segments, info) where info carries detected language and duration.
    """
    # Imported lazily so --skip-transcribe / --help stay fast and don't drag
    # torch/ctranslate2 in if the user only wants to download.
    from faster_whisper import WhisperModel

    print(
        f"[whisper] loading model={model_name!r} (device=cpu, compute=int8) ...",
        file=sys.stderr,
    )
    # int8 keeps it usable on plain CPU. If you have a GPU, swap to
    # device="cuda", compute_type="float16".
    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    print(f"[whisper] transcribing {audio_path.name} ...", file=sys.stderr)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,             # None = auto-detect
        vad_filter=True,               # skip silent stretches -> cleaner output
        beam_size=5,
    )

    segments = [
        TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
        for s in segments_iter
    ]
    meta = {
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration": float(info.duration),
    }
    print(
        f"[whisper] done: language={meta['language']} "
        f"p={meta['language_probability']:.2f} duration={meta['duration']:.1f}s "
        f"segments={len(segments)}",
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
        description="Download a YouTube video's audio (yt-dlp) and transcribe it (faster-whisper).",
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
            "faster-whisper model name (tiny, base, small, medium, large-v3, "
            "distil-large-v3). Larger = better & slower. Default: %(default)s"
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
        help=f"yt-dlp audio format (mp3, m4a, wav, opus, ...). Default: %(default)s",
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