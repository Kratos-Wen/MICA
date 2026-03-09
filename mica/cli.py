"""CLI entrypoint for the modular open-source MICA package."""

from __future__ import annotations

import argparse
from pathlib import Path

from mica.config import load_config
from mica.runtime.live_runner import run_camera
from mica.runtime.offline_runner import run_video
from mica.runtime.sources import discover_yolo_weights


def _default_kb_path() -> str:
    return str(Path(__file__).resolve().parent / "resources" / "kb.example.json")


def _apply_cli_ablation_flags(cfg, args) -> None:
    ablation = cfg.raw.setdefault("ablation", {})
    if args.disable_depth_context:
        ablation["use_depth_guided_context"] = False
    if args.disable_state_graph_expert:
        ablation["use_state_graph_expert"] = False
    if args.disable_retrieval_expert:
        ablation["use_retrieval_expert"] = False
    if args.disable_asf:
        ablation["use_asf"] = False
    if args.disable_mica_core:
        ablation["use_mica_core"] = False
    if args.agent_topology:
        cfg.raw.setdefault("agent", {})["topology"] = args.agent_topology


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("MICA modular open-source package")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video", type=str, help="Offline video path for test data runs.")
    source_group.add_argument("--camera", type=int, help="Camera index for live runs.")

    parser.add_argument("--yolo-weights", type=str, default=None, help="YOLO weights (.pt).")
    parser.add_argument("--kb", type=str, default=None, help="KB JSON path.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    parser.add_argument("--device", type=str, default="cpu", help="YOLO device or 'cpu'.")
    parser.add_argument("--interactive", action="store_true", help="Enable console HITL prompts.")
    parser.add_argument(
        "--agent-topology",
        type=str,
        default=None,
        choices=["mica", "shared", "central", "hier", "debate"],
        help="Reasoning topology for MICA-core ablations.",
    )
    parser.add_argument("--disable-depth-context", action="store_true", help="Disable depth-guided context filtering.")
    parser.add_argument("--disable-state-graph-expert", action="store_true", help="Disable the KB state-graph expert.")
    parser.add_argument("--disable-retrieval-expert", action="store_true", help="Disable the gallery retrieval expert.")
    parser.add_argument("--disable-asf", action="store_true", help="Disable Adaptive Step Fusion.")
    parser.add_argument("--disable-mica-core", action="store_true", help="Disable MICA-core question answering.")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    _apply_cli_ablation_flags(cfg, args)
    kb_path = args.kb or _default_kb_path()
    yolo_weights = discover_yolo_weights(args.yolo_weights)
    topology = args.agent_topology or str(cfg.agent.get("topology", "mica"))

    if args.video:
        run_dir = run_video(
            video_path=args.video,
            yolo_weights=yolo_weights,
            kb_path=kb_path,
            cfg=cfg,
            device=args.device,
            interactive=args.interactive,
            topology=topology,
        )
    else:
        run_dir = run_camera(
            camera_index=int(args.camera),
            yolo_weights=yolo_weights,
            kb_path=kb_path,
            cfg=cfg,
            device=args.device,
            interactive=args.interactive,
            topology=topology,
        )
    print(f"[DONE] Artifacts saved to: {run_dir}")
