# LLM Review Tracer

A tool for paper authors and conference organisers to watermark submitted PDFs so that AI-generated peer reviews can be detected. Each reviewer receives a uniquely watermarked copy of the paper containing invisible vocabulary instructions. If a reviewer uses an LLM to write their review, the LLM follows those instructions and leaves a detectable synonym fingerprint in the submitted review text.

## How it works

1. **Embed** — a per-reviewer token is derived from a secret salt and a paper ID. The token encodes 20 synonym preferences (e.g. "work" vs "paper"). These preferences are injected as invisible white text behind figures and at periodic page positions in the PDF.
2. **Review** — the reviewer reads the PDF, optionally feeds it to an LLM. The LLM sees the vocabulary instructions and (unknowingly) follows them.
3. **Detect** — the organiser runs detection on the submitted review. Synonym frequencies are decoded back into a token and matched against the registry using fuzzy Hamming matching (threshold ≤ 6/20 bits).

## Setup

### Requirements

- Python 3.8+
- [PyMuPDF](https://pymupdf.readthedocs.io/) (`pip install pymupdf`)

```bash
pip install pymupdf
```

The bundled `NotoSans-Regular.ttf` is required for cross-platform Unicode support (included in this repo — no system fonts needed).

### Installation

```bash
git clone <repo-url>
cd LLM-Reviewer
pip install pymupdf
```

No further configuration is needed. On first run a secret salt is generated automatically and saved to `.watermark_salt` — **keep this file secret and back it up**. Detection is impossible without it.

---

## Usage

### 1. Watermark a paper

```bash
python pipeline.py embed \
  --input-pdf  paper.pdf \
  --output-pdf paper-wm.pdf \
  --paper-id   S660
```

Send `paper-wm.pdf` to the reviewer. Keep `paper.pdf` private.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--input-pdf` | `test.pdf` | Original PDF to watermark |
| `--output-pdf` | `test-wm.pdf` | Output watermarked PDF |
| `--paper-id` | `S660` | Unique identifier for this paper |
| `--step` | `3` | Insert per-page watermark every N pages. Higher = stealthier (try `6` or `9` to reduce detectability by AI models) |
| `--zwc-font` | auto | Path to a TTF font file. Defaults to `NotoSans-Regular.ttf` in the script directory |
| `--registry` | `wm_registry.csv` | Path to the token registry CSV |
| `--salt-file` | `.watermark_salt` | Path to the secret salt file |
| `--skip-verify` | off | Skip post-embed verification |

**Multiple reviewers for the same paper:**

Use a different `--paper-id` per reviewer copy so each gets a unique token:

```bash
python pipeline.py embed --input-pdf paper.pdf --output-pdf paper-wm-r1.pdf --paper-id S660-R1
python pipeline.py embed --input-pdf paper.pdf --output-pdf paper-wm-r2.pdf --paper-id S660-R2
python pipeline.py embed --input-pdf paper.pdf --output-pdf paper-wm-r3.pdf --paper-id S660-R3
```

---

### 2. Detect an AI-generated review

Save the submitted review as a plain text file, then run:

```bash
python pipeline.py detect --review review.txt
```

**Example output — detected:**
```
============================================================
  [DETECTED] LLM-generated review found!
  Paper ID   : S660
  Token      : 19E652AF  (decoded=252AF, hamming=1/20)
  Confidence : 19/20 bits matched  (95%)
  Pairs seen : 17/20  (threshold=6)
============================================================
```

**Example output — not detected:**
```
============================================================
  [NOT DETECTED] best match hamming=8/20  (triggered=14/20, threshold=6)
============================================================
```

**Options:**

| Flag | Description |
|---|---|
| `--review` | Path to the review text file (required) |
| `--registry` | Path to the token registry CSV (default: `wm_registry.csv`) |
| `--strip-ch1` | Simulate an attacker who found and removed the visible instruction blocks. Tests whether the ZWC channel alone survives. |

---

## Stealth tuning

The `--step` parameter controls how many per-page watermark copies are inserted alongside the per-figure copies. Lower values mean more copies (stronger signal, easier to detect by inspection); higher values mean fewer copies (stealthier, but still detectable):

| `--step` | Per-page copies (20-page paper) | Notes |
|---|---|---|
| `1` | 20 | Every page — maximum signal, likely flagged by Claude/GPT-4o |
| `3` | 7 | Default — good balance |
| `6` | 4 | Stealthy — tested: not flagged, still detected |
| `9` | 3 | Very stealthy — borderline signal, use only with many figures |

Figure-placement copies are always included regardless of `--step`.

---

## Files

| File | Description |
|---|---|
| `pipeline.py` | Main script — embed and detect |
| `NotoSans-Regular.ttf` | Bundled font (required for cross-platform Unicode support) |
| `.watermark_salt` | Secret salt — **do not share or lose** (auto-generated on first run) |
| `wm_registry.csv` | Token registry — maps tokens to paper IDs and bit strings |

---

## Notes

- The watermark is invisible: white text placed behind page content and figures.
- The detection is fuzzy: up to 6/20 synonym bits may be wrong (LLM paraphrasing, rare words not appearing) and the review will still be detected.
- A human-written review scores ~8–10/20 Hamming distance from any token — well above the detection threshold of 6.
- The scheme works on text-extractable PDFs only. OCR-based processing strips all watermarks.
