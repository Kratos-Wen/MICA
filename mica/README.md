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
- This repository already includes `dataset/components.json`, `dataset/KB.json`, and `checkpoint/best.pt`.

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

Bundled in this repository:

- `checkpoint/best.pt`
- `dataset/components.json`
- `dataset/KB.json`

Prepare the following input before testing:

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
  --kb dataset/components.json \
  --yolo-weights checkpoint/best.pt \
  --config mica/resources/config.example.yaml \
  --device cpu
```

### Live Camera

```bash
python -m mica \
  --camera 0 \
  --kb dataset/components.json \
  --yolo-weights checkpoint/best.pt \
  --config mica/resources/config.example.yaml \
  --device cpu \
  --interactive
```

If `--kb` is omitted, the CLI first uses `dataset/components.json`. If `--yolo-weights` is omitted, it first uses `checkpoint/best.pt`.

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
  --kb dataset/components.json \
  --yolo-weights checkpoint/best.pt \
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

## Gear8 Dataset

The Gear8 dataset used in this work will be released on Hugging Face.

- Download link: **[Coming soon](https://huggingface.co/datasets)**
