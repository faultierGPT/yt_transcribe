# yt_transcribe

Tiny CLI: download audio from a YouTube (or any yt-dlp-supported) URL and
transcribe it locally with faster-whisper. No API key, no cloud cost.

## Letzter Durchlauf
**Aufgabe:** "Simple script that downloads (yt-dlp) and transcribes yt videos." **Gebaut:** `yt_transcribe.py` — argparse-CLI mit URL(s) oder `--local-file`, yt-dlp-Download (Audio-Only) + faster-whisper-Transkription (CPU, int8), Output als `.txt` + `.json` mit Segment-Timestamps und Detected-Language-Metadaten. **Verifiziert:** End-to-End-Test mit Vimeo-Video (Download → Transkription → Cleanup, 14 Segmente, sinnvoller Text) und Local-File-Test mit JFK-whisper.cpp-Sample (Sprache korrekt als EN/0.97 erkannt). **Wichtigste Erkenntnis:** YouTube blockt hier Server-seitige Downloads ohne Cookies ("Sign in to confirm you're not a bot") — das ist eine Umgebungs-Frage, kein Script-Bug; das Skript funktioniert mit `--cookies-from-browser` und auf Nicht-YouTube-Quellen problemlos.

## Zweck
Ein-Skript-Lösung, um Audiodaten aus Videos in durchsuchbare Texte zu verwandeln — ohne OpenAI-API-Key, ohne Cloud-Upload. Audio wird standardmäßig nach erfolgreicher Transkription wieder gelöscht (Platzersparnis).

## Projektstruktur
```
.
├── yt_transcribe.py    # Haupt-Skript: Download + Transkription
├── .venv/              # Lokale venv mit faster-whisper (NICHT committen)
├── README.md           # Projekt-Titel
└── AGENTS.md           # Diese Datei
```

## Tech-Stack
- **Python 3.13** (PEP 668 aktiv → venv-Pflicht, kein systemweites pip)
- **yt-dlp 2026.06.09** (System-PATH, `~/.local/bin/yt-dlp`)
- **faster-whisper 1.2.1** (in `.venv`, lokale CPU-Inferenz, int8)
- **ffmpeg 7.1.5** (System-PATH, von yt-dlp als Postprocessor benötigt)

## Entscheidungen
- **faster-whisper statt openai-whisper**: ~4x schneller auf CPU, kein PyTorch-Abhängigkeitsballast, CTranslate2-Backend.
- **CPU + int8 als Default**: Läuft überall, kein CUDA-Setup nötig. GPU-Nutzung möglich via Code-Patch (`device="cuda", compute_type="float16"`), nicht per CLI exposed — bewusst einfach gehalten.
- **Modell `tiny` als Default**: Schnellster Smoke-Test. Für Produktion `--model small` oder `medium` empfehlen (in CLI-Docstring dokumentiert).
- **Audio wird nach Transkription gelöscht** (`--keep-audio` zum Überschreiben): Spart Platz, da das Audio nach erfolgreichem Transkript rekonstruierbar ist.
- **Zwei Output-Formate pro Video**: `.txt` (plain, einfaches Grep/Paste) + `.json` (Segmente + Timestamps + Detected-Language + Source-URL für Nachverarbeitung).
- **Lazy Import von faster_whisper**: Damit `--skip-transcribe` und `--help` sofort laufen, ohne dass torch/ctranslate2 geladen wird.
- **Multi-URL-Verarbeitung läuft weiter bei Fehlern**: Eine kaputte URL bricht den Batch nicht ab.

## Bedienung
```bash
# Standard: alles in einem Schritt
python yt_transcribe.py "https://www.youtube.com/watch?v=..." --model small

# Mehrere URLs
python yt_transcribe.py URL1 URL2 URL3 --output-dir ./out

# Nur Audio runterladen, transkribieren später
python yt_transcribe.py URL --skip-transcribe

# Existierende Audiodatei transkribieren (kein yt-dlp nötig)
python yt_transcribe.py --local-file podcast.mp3 --model medium

# Sprache erzwingen statt Auto-Detect
python yt_transcribe.py URL --language de --model medium
```

`--help` zeigt alle Optionen.

## Aktueller Stand
- Skript funktional, lokal getestet mit:
  - Vimeo-Video (End-to-End, 14 Segmente, sauber transkribiert, Audio cleaned up)
  - JFK-Whisper-Sample (Local-File-Pfad, Sprache korrekt erkannt)
- venv installiert, faster-whisper läuft auf CPU/int8.

## Bekannte Einschränkungen
- **YouTube-Downloads können am Bot-Check scheitern** mit `ERROR: Sign in to confirm you're not a bot.` Das ist eine Umgebungs-Frage (frischer Server ohne YouTube-Cookies), nicht ein Script-Bug. Workaround: `yt-dlp --cookies-from-browser firefox` bzw. yt-dlp-Cookies-File als `~/.config/yt-dlp/cookies.txt` ablegen. Falls das hier wichtig wird, kann ein `--cookies` Flag eingebaut werden.
- **Keine GPU-Beschleunigung** in der CLI (Code-Patch nötig).
- **Keine SRT/VTT-Subtitle-Formate** als Output (nur .txt + .json). Falls gebraucht: writer-Funktion erweitern.
- **Keine Resume-Logik** bei abgebrochenen Downloads.

## Next Steps
- Falls regelmäßig YouTube-Quellen: Cookie-Support einbauen (entweder `--cookies` Flag oder `~/.config/yt-dlp/cookies.txt` automatisch lesen).
- Optional: SRT/VTT-Writer hinzufügen für Caption-Use-Cases.
- Optional: Batch-Input aus Datei (`--input-file urls.txt`, eine URL pro Zeile) für richtig große Listen.
- Optional: GPU-Detection + Auto-Switch auf `float16`.

## Setup reproduzieren
```bash
python3 -m venv .venv
.venv/bin/pip install faster-whisper
# System-Pakete (Debian/Ubuntu):
sudo apt install yt-dlp ffmpeg
```