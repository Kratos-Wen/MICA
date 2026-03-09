"""Artifact logging for offline and live runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

from mica_glasses_open.runtime.kb_utils import ensure_parent


class RunArtifacts:
    """Own the run directory and structured logs."""

    def __init__(self, run_dir: Path, meta: Dict[str, Any]) -> None:
        self.run_dir = Path(run_dir)
        self.meta = dict(meta)
        ensure_parent(self.run_dir / "meta.json")
        (self.run_dir / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

        self.jsonl_path = self.run_dir / "iterations.jsonl"
        self.csv_path = self.run_dir / "summary.csv"
        self.feedback_path = self.run_dir / "feedback_log.jsonl"
        self.annotated_video_path: Optional[Path] = None

        self._jsonl = self.jsonl_path.open("w", encoding="utf-8")
        self._feedback = self.feedback_path.open("a", encoding="utf-8")
        self._csv_file = self.csv_path.open("w", encoding="utf-8", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(
            [
                "iter",
                "frame_index",
                "num_raw",
                "num_fused",
                "nearest",
                "stable",
                "s_step",
                "s_conf",
                "r_step",
                "r_conf",
                "fused",
                "f_conf",
                "chosen",
                "w_s",
                "w_r",
            ]
        )

    @classmethod
    def create(cls, cfg, source_label: str, meta: Dict[str, Any]) -> "RunArtifacts":
        save_root = Path(cfg.runlog.get("save_dir", "runs_open"))
        if source_label.startswith("camera://"):
            run_name = source_label.replace("camera://", "camera_")
        else:
            run_name = Path(source_label).stem
        run_dir = save_root / run_name
        return cls(run_dir=run_dir, meta=meta)

    def build_annotated_video_path(self, stem: str) -> Path:
        path = self.run_dir / f"{stem}_annotated.mp4"
        self.annotated_video_path = path
        return path

    def save_snapshot(self, image, iter_index: int, frame_index: int) -> None:
        import cv2

        path = self.run_dir / f"vis_iter_{iter_index}_frame_{frame_index}.jpg"
        cv2.imwrite(str(path), image)

    def log_iteration(self, decision) -> None:
        record = {
            "iter": decision.iter_index,
            "frame_index": decision.frame_index,
            "num_raw": len(decision.perception.raw_detections),
            "num_fused": len(decision.perception.fused_detections),
            "nearest_index": decision.perception.nearest_index,
            "stable": bool(decision.is_stable),
            "s_step": decision.step_prediction.state_step,
            "s_conf": decision.step_prediction.state_conf,
            "r_step": decision.step_prediction.retrieval_step,
            "r_conf": decision.step_prediction.retrieval_conf,
            "fused": decision.step_prediction.fused_step,
            "f_conf": decision.step_prediction.fused_conf,
            "chosen": decision.step_prediction.chosen,
            "w_s": decision.step_prediction.weights.get("s", 0.0),
            "w_r": decision.step_prediction.weights.get("r", 0.0),
            "detections": decision.perception.fused_detections,
            "rel_dets": decision.perception.relevant_detections,
            "signature": decision.perception.signature,
        }
        self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._csv.writerow(
            [
                decision.iter_index,
                decision.frame_index,
                len(decision.perception.raw_detections),
                len(decision.perception.fused_detections),
                decision.perception.nearest_index,
                int(decision.is_stable),
                decision.step_prediction.state_step,
                f"{decision.step_prediction.state_conf:.3f}",
                decision.step_prediction.retrieval_step,
                f"{decision.step_prediction.retrieval_conf:.3f}",
                decision.step_prediction.fused_step,
                f"{decision.step_prediction.fused_conf:.3f}",
                decision.step_prediction.chosen,
                f"{decision.step_prediction.weights.get('s', 0.0):.3f}",
                f"{decision.step_prediction.weights.get('r', 0.0):.3f}",
            ]
        )

    def log_feedback(self, payload: Dict[str, Any]) -> None:
        self._feedback.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def finalize(self) -> None:
        try:
            self._jsonl.close()
            self._csv_file.close()
            self._feedback.close()
        finally:
            manifest = {
                "source": self.meta.get("source"),
                "artifacts": {
                    "iterations": str(self.jsonl_path),
                    "summary": str(self.csv_path),
                    "feedback_log": str(self.feedback_path),
                },
            }
            if self.annotated_video_path is not None:
                manifest["artifacts"]["annotated_video"] = str(self.annotated_video_path)
            (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
