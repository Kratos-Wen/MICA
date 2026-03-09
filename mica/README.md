<div align="center">
  <h1>MICA: Multi-Agent Industrial Coordination Assistant</h1>
  <p>Paper-aligned modular implementation for wearable industrial assistance</p>
</div>

## Overview

This repository provides a clean, modular implementation of the MICA pipeline described in the accompanying paper.
The codebase is organized around the three core modules in the method:

- `Depth-guided Object Context Extraction`
- `Adaptive Assembly Step Recognition`
- `MICA-core`

The runtime supports both offline video evaluation and live camera inference.

## Path Configuration Notice

- Every path written as `/path/to/...` in this README is a placeholder.
- Replace these placeholders with paths on your own machine before running.
- If `--yolo-weights` is omitted, the CLI looks for `best.pt` in the working directory.

## Setup

### 1. Create an Environment

Create and activate a clean conda environment, then install dependencies:

```bash
conda create -n mica python=3.10 -y
conda activate mica
pip install -U pip
pip install -r requirements.txt
```

### 2. Required Assets

Prepare the following assets before testing:

- A YOLO checkpoint (`.pt`)
- A knowledge base JSON file
- An offline video for test runs, or a valid camera index for live runs

Optional assets:

- A retrieval gallery organized by step folders
- A local Ollama-compatible endpoint for MICA-core question answering

If `gallery.root` is left empty in the config, the runtime skips gallery indexing and still runs.

## Run

From the repository root:

### Offline Video

```bash
python -m mica \
  --video /path/to/video.mp4 \
  --kb /path/to/kb.json \
  --yolo-weights /path/to/best.pt \
  --config resources/config.example.yaml \
  --device cpu
```

### Live Camera

```bash
python -m mica \
  --camera 0 \
  --kb /path/to/kb.json \
  --yolo-weights /path/to/best.pt \
  --config resources/config.example.yaml \
  --device cpu \
  --interactive
```

If `--kb` is omitted, the example knowledge base in `resources/kb.example.json` is used.

## Runtime Controls

Live mode supports the following keyboard controls:

- `Q`: quit
- `P` or `Space`: pause or resume
- `F`: request a console feedback or QA prompt on the next stable step
- `H`: toggle the help overlay

## Ablation

The CLI exposes component-level ablations directly:

- `--disable-depth-context`
- `--disable-state-graph-expert`
- `--disable-retrieval-expert`
- `--disable-asf`
- `--disable-mica-core`
- `--agent-topology {mica,shared,central,hier,debate}`

Example:

```bash
python -m mica \
  --video /path/to/video.mp4 \
  --kb /path/to/kb.json \
  --yolo-weights /path/to/best.pt \
  --disable-retrieval-expert
```

## Output

Each run writes structured artifacts to the configured output directory, including:

- `iterations.jsonl`
- `summary.csv`
- `feedback_log.jsonl`
- `manifest.json`
- optional annotated video
- persistent ASF weights
