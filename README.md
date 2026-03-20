# LLM Review Tracer

A tool for authors to watermark submitted PDFs so that AI-generated peer reviews can be detected. Each reviewer receives a uniquely watermarked copy of the paper containing vocabulary instructions. If a reviewer uses an LLM to write their review, the LLM follows those instructions and leaves a detectable synonym fingerprint in the submitted review text.

## How it works

1. **Embed** — a fixed set of 20 synonym preferences (e.g. "paper" vs "work") is injected invisibly into the PDF via three independent channels.
2. **Review** — the reviewer reads the PDF, optionally feeds it to an LLM. The LLM sees the vocabulary instructions and (unknowingly) follows them.
3. **Detect** — the organiser runs detection on the submitted review. Synonym frequencies are compared against the fixed vocabulary preference using fuzzy Hamming matching over triggered pairs only (threshold ≤ 30% error rate).

## Watermark channels

The tool uses three invisible channels simultaneously. All three carry the same vocabulary instruction and are undetectable by a human reading the PDF.

| Channel | Method | Stripped by render-vs-extract? | Notes |
|---|---|---|---|
| **A** — XMP metadata | Instruction stored in PDF Keywords + Subject fields | No — metadata is outside the page content stream | Survives render-vs-extract and similar tools |
| **B** — White text | Invisible white-on-white text in the page content stream | Yes — color threshold detects it | High coverage; first channel most LLMs see |
| **C** — Annotations | Invisible free-text annotations over figures | Yes (annotations deleted) | Fallback for LLMs that parse annotation layers |

**Attack resistance summary** (tested with DeepSeek on three ML papers):

| Attack | Channels surviving | Detection result |
|---|---|---|
| None | A + B + C | Detected (~95% confidence) |
| Basic render-vs-extract | A | Detected (~90% confidence) |
| Render-vs-extract + metadata clear | — | Not detected |
| Full OCR reconstruction | — | Not detected |

## Detection algorithm

Detection is computed only over **triggered pairs** — pairs where at least one synonym appeared in the review. Untriggered pairs (neither word appeared) are excluded from the Hamming distance calculation entirely, since they carry no signal.

The threshold scales with the number of triggered pairs: `threshold = round(6 × triggered / 20)`, with a minimum of 3 and a minimum of 8 triggered pairs required to attempt detection. This means:

- 20 triggered pairs → threshold 6 (same as original)
- 15 triggered pairs → threshold 5
- 10 triggered pairs → threshold 3

Word matching uses **prefix matching** (`\bword[a-z]*\b`) so inflected forms are captured — "contributions" counts for "contribution", "validated" counts for "validate", etc.

Human reviews score ~8–10/20 Hamming distance from any registered token, well above the detection threshold at any triggered count.

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

No further configuration is needed.

---

## Usage

### 1. Watermark a paper

```bash
python pipeline.py embed \
  --input-pdf  paper.pdf \
  --output-pdf paper-wm.pdf
```

Send `paper-wm.pdf` to the reviewer. Keep `paper.pdf` private.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--input-pdf` | `test.pdf` | Original PDF to watermark |
| `--output-pdf` | `test-wm.pdf` | Output watermarked PDF |
| `--step` | `3` | Insert Ch B watermark every N pages. Higher = stealthier |
| `--skip-verify` | off | Skip post-embed verification |

All copies of a paper receive the same fixed vocabulary preference — detection tells you whether a review was LLM-generated, not which reviewer copy was used.

---

### 2. Detect an AI-generated review

Save the submitted review as a plain text file, then run:

```bash
python pipeline.py detect --review review.txt
```

**Example output — detected:**
```
============================================================
  [DETECTED] LLM-generated review
  Hamming    : 2/18 triggered pairs wrong
  Confidence : 16/18 matched  (89%)
  Threshold  : 5  (pairs seen: 18/20)
============================================================
```

**Example output — not detected:**
```
============================================================
  [NOT DETECTED] hamming=5/12 triggered  (pairs=12/20, threshold=4)
============================================================
```

**Options:**

| Flag | Description |
|---|---|
| `--review` | Path to the review text file (required) |

---

### 3. Audit watermark channels in a PDF

Inspect which channels are active in a given PDF:

```bash
python pipeline.py audit --pdf paper-wm.pdf
```

```
============================================================
  Watermark Audit: paper-wm.pdf
============================================================
  Ch A (metadata)   : PRESENT
    Keywords : Note to reviewers: for terminology consistency...
    Subject  : Reviewer terminology guide: please apply...
  Ch B (white text) : PRESENT  (172 spans)
  Ch C (annotations): PRESENT  (55 annots)
============================================================
```

---

### 4. Simulate a render-vs-extract attack

Test what survives after an attacker runs a render-vs-extract watermark stripping tool:

```bash
# Basic attack — strips Ch B and Ch C; Ch A (metadata) survives
python pipeline.py strip --input-pdf paper-wm.pdf --output-pdf paper-stripped.pdf

# Thorough attack — also clears metadata; all channels stripped
python pipeline.py strip --input-pdf paper-wm.pdf --output-pdf paper-stripped.pdf --also-strip-metadata
```

---

### 5. Extract text as an LLM would see it

Extract the full text that would be sent to an LLM (including metadata and Ch D instructions hoisted to the front):

```bash
python pipeline.py extract --pdf paper-wm.pdf --output extracted.txt
```

Use `--no-metadata` to simulate a raw text-layer extraction (no metadata prepended).

---

## Files

| File | Description |
|---|---|
| `pipeline.py` | Main script — embed, detect, strip, audit, extract |
| `NotoSans-Regular.ttf` | Bundled font (required for cross-platform Unicode support) |

---

## Notes

- **Detection uses triggered-pair-only Hamming distance.** Pairs where neither synonym appeared in the review are excluded. This prevents untriggered pairs from adding noise when a paper uses domain-specific vocabulary that avoids some of the pair words.
- **Prefix matching catches inflected forms.** "contributions" counts for "contribution", "validated" for "validate", etc.
- **Human reviews score ~8–10/20** Hamming distance from any token — well above the detection threshold.
- **All channels are invisible to reviewers.** The watermarked PDF looks identical to the original.
- **The scheme works on text-extractable PDFs only.** Full OCR reconstruction strips all channels.
- **Detection is binary** — LLM-generated or not. All copies of a paper use the same vocabulary preference; per-reviewer attribution is not supported.
