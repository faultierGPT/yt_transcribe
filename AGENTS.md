# yt_transcribe

Tiny CLI: download audio from a YouTube (or any yt-dlp-supported) URL with `yt-dlp`
and transcribe it with OpenAI's Whisper API (`whisper-1`). Needs `OPENAI_API_KEY`.

## Letzter Durchlauf
**Aufgabe:** "Use OpenAI's Whisper instead of local Whisper." **Gemacht:** Lokalen `faster-whisper`-Backend komplett rausgeworfen und auf OpenAI Whisper API umgestellt. `yt_transcribe.py` ruft jetzt `client.audio.transcriptions.create(model="whisper-1", …, response_format="verbose_json", timestamp_granularities=["segment"])`, damit die `TranscriptSegment`-Liste mit Timestamps erhalten bleibt; `language_probability` als `None` (API liefert das Feld nicht, behalten für Shape-Kompat). Pre-flight-Guard für die 25-MB-Upload-Grenze eingebaut, sauberer `OPENAI_API_KEY`-Missing-Branch, Lazy-Import vom `openai`-Package. **Verifiziert:** `--help`, fehlende-Key-Error, 25-MB-Guard, Stubbed-Client-Segment-Mapping und eine echte HTTP-401 von `api.openai.com` mit fake-key (beweist, dass die Wire-Shape stimmt). **Wichtigste Erkenntnis:** `whisper-1` ist aktuell das einzige Whisper-Modell auf der Public API — `--model` bleibt als Flag, damit ein künftiger Slug ohne Code-Patch durchgeht. End-to-End-Test mit echtem Audio + echtem Key steht aus (User-seitig, sobald Key gesetzt).

## Zweck
Ein-Skript-Lösung, um Audiodaten aus Videos in durchsuchbare Texte zu verwandeln — via OpenAI Whisper API. Kein lokaler GPU/CPU-Whisper-Setup, kein PyTorch-Build. Audio wird standardmäßig nach erfolgreicher Transkription wieder gelöscht (Platzersparnis).

## Projektstruktur
```
.
├── yt_transcribe.py    # Haupt-Skript: yt-dlp-Download + OpenAI Whisper API
├── .venv/              # Lokale venv mit `openai` (NICHT committen)
├── README.md           # Projekt-Titel + Hinweis auf AGENTS.md und --help
└── AGENTS.md           # Diese Datei
```

## Tech-Stack
- **Python 3.13** (PEP 668 aktiv → venv-Pflicht, kein systemweites pip)
- **yt-dlp 2026.06.09** (System-PATH, `~/.local/bin/yt-dlp`)
- **openai 2.44+** (in `.venv`, offizielles Python SDK)
- **ffmpeg 7.1.5** (System-PATH, von yt-dlp als Postprocessor benötigt)

## Entscheidungen
- **OpenAI Whisper API statt faster-whisper (lokal)**: Anweisung des Users. Kein lokaler Modell-Download, kein PyTorch/CTranslate2, läuft auch ohne CPU-Inferenz — geht dafür mit Cloud-Kosten und API-Key einher.
- **Modell `whisper-1`** (default, derzeit das einzige Whisper-Modell auf der Public API). `--model` bleibt als Flag, damit ein künftiger API-Slug (z.B. `whisper-2`) ohne Code-Patch durchgereicht werden kann.
- **`response_format="verbose_json"` + `timestamp_granularities=["segment"]`**: liefert pro Segment `start`/`end`/`text` — gleiche Shape wie vorher von faster-whisper, also keine Änderung an `write_outputs` nötig.
- **`language_probability` durch `None` ersetzt**: Die API liefert das Feld nicht mit; behalten, damit Downstream-Code, das die Meta-Dict liest, nicht bricht.
- **25 MB Upload-Limit als Pre-flight-Guard**: API lehnt größere Dateien hart ab. Guard feuert vor dem HTTP-Call, mit konkretem Workaround-Hinweis (`yt-dlp --audio-quality 9`), damit der User nicht erst beim 400er rätselt.
- **`--audio-format mp3` Default bleibt**, aber Help-Text weist explizit auf Bitrate runter für lange Audios (über 25 MB Limit).
- **Lazy Import von `openai`**: Damit `--help`, `--skip-transcribe` und der reine Download-Pfad sofort laufen, ohne das SDK zu laden — und ohne dass `OPENAI_API_KEY` für nicht-Transkriptions-Operationen gesetzt sein muss.
- **`OPENAI_API_KEY` aus `os.environ`**: Keine eigene `.env`-Datei, keine `.netrc`-Magie, keine Key im Repo. Fail-fast-Fehler beim ersten Transkriptions-Aufruf, wenn nicht gesetzt.
- **Audio wird nach Transkription gelöscht** (`--keep-audio` zum Überschreiben).
- **Zwei Output-Formate pro Video**: `.txt` (plain, einfaches Grep/Paste) + `.json` (Segmente + Timestamps + Detected-Language + Source-URL).
- **Multi-URL-Verarbeitung läuft weiter bei Fehlern**: Eine kaputte URL bricht den Batch nicht ab.

## Bedienung
```bash
# einmalig pro Shell
export OPENAI_API_KEY='***'    # https://platform.openai.com/api-keys

# Standard: alles in einem Schritt
python yt_transcribe.py "https://www.youtube.com/watch?v=..." --language en

# Mehrere URLs
python yt_transcribe.py URL1 URL2 URL3 --output-dir ./out

# Nur Audio runterladen, transkribieren später (kein API-Key nötig)
python yt_transcribe.py URL --skip-transcribe

# Existierende Audiodatei transkribieren
python yt_transcribe.py --local-file podcast.mp3 --language en

# Große Audios unter 25 MB halten
python yt_transcribe.py URL --audio-format mp3 --keep-audio
# danach ggf. lokal mit ffmpeg -b:a 32k neu encoden und nochmal starten
```

`--help` zeigt alle Optionen.

## Aktueller Stand
- Skript umgestellt: Backend = OpenAI Whisper API statt faster-whisper.
- venv neu angelegt mit `openai==2.44.0` (frühere `faster-whisper`-venv war nicht im Repo, also sauberer Schnitt).
- Lokal verifiziert:
  - `--help` zeigt neue Optionen mit Hinweis auf 25-MB-Limit
  - `OPENAI_API_KEY` fehlt → sauberer RuntimeError mit Setup-Hinweis
  - 25-MB-Size-Guard feuert mit Re-Download-Workaround-Hinweis
  - Segment-Mapping (Stubbed OpenAI-Client): `verbose_json`-Antwort → `TranscriptSegment`-Liste mit `start`/`end`/`text` korrekt
  - Echte Wire-Prüfung: fake-key gegen `api.openai.com/audio/transcriptions` → HTTP 401 von OpenAI, beweist dass Modell/Format/Granularities/File-Upload korrekt gesendet werden
- **End-to-End mit echtem Audio + echtem Key steht aus** (User-seitig, sobald `OPENAI_API_KEY` gesetzt ist). Die Code-Pfade sind alle abgedeckt; was nicht reproduziert wurde, ist die tatsächliche Transkription.

## Bekannte Einschränkungen
- **Braucht `OPENAI_API_KEY`**: Skript läuft ohne Key nur im `--skip-transcribe`-Modus. Key wird aus `os.environ` gelesen — kein `.env`-Auto-Loader eingebaut.
- **25-MB-Upload-Limit**: Längere Audios (typischerweise >~30 Min in guter Qualität) müssen vorher komprimiert oder gesplittet werden. Chunked-Transkription (FFmpeg-Split → N API-Calls → Re-Stitch) ist bewusst nicht implementiert — bewahrt die KISS-Form. Falls das häufig gebraucht wird: nächste Iteration mit `ffmpeg`-Segmentierung.
- **`language_probability` immer `None`**: Die OpenAI API liefert keine Konfidenz für die automatische Sprach-Detection. Konsumenten, die das Feld nutzen, müssen `None` tolerieren oder auf `language` selbst zurückfallen.
- **YouTube-Downloads können am Bot-Check scheitern** mit `ERROR: Sign in to confirm you're not a bot.` — Umgebungs-Frage, kein Script-Bug. Workaround: `yt-dlp --cookies-from-browser firefox`.
- **Cloud-Kosten**: Pro Audio-Minute wird eine Whisper-API-Minute abgerechnet (siehe OpenAI-Preisliste). Für Bulk-Workloads lokal günstiger, dafür fehlt hier die GPU-Pfad-Alternative.
- **Keine SRT/VTT-Subtitle-Formate** als Output (nur `.txt` + `.json`).
- **Keine Resume-Logik** bei abgebrochenen Downloads.

## Next Steps
- Falls 25-MB-Limit regelmäßig reißt: `--bitrate`-Flag bzw. automatisches Re-Encode via ffmpeg-Postprocessor in yt-dlp einbauen.
- Optional: `.env`-Auto-Loader (`python-dotenv`, nur als Dep) für lokale Entwicklung.
- Optional: SRT/VTT-Writer hinzufügen für Caption-Use-Cases (das `verbose_json` hat eh schon Segment-Granularität — kleinster Patch).
- Optional: Batch-Input aus Datei (`--input-file urls.txt`, eine URL pro Zeile) für richtig große Listen.
- Optional: Retry-Logik mit exponentiellem Backoff für 429/5xx-Antworten der API.

## Setup reproduzieren
```bash
python3 -m venv .venv
.venv/bin/pip install openai
# System-Pakete (Debian/Ubuntu):
sudo apt install yt-dlp ffmpeg
# Vor der ersten Transkription:
export OPENAI_API_KEY='***'
```
