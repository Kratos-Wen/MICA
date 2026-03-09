"""Console interaction helpers for stable-step feedback and QA."""

from __future__ import annotations


def handle_stable_console_interaction(pipeline, decision, artifacts, topology: str = ""):
    print(
        f"[HITL] Stable at iter={decision.iter_index}. "
        f"Predicted: fused={decision.step_prediction.fused_step} ({decision.step_prediction.fused_conf:.2f}), "
        f"s={decision.step_prediction.state_step}/{decision.step_prediction.state_conf:.2f}, "
        f"r={decision.step_prediction.retrieval_step}/{decision.step_prediction.retrieval_conf:.2f}"
    )
    user_input = input(
        "[HITL] ENTER accept | S1/S2/... correct | skip ignore | qa ask a question: "
    ).strip()
    if not user_input:
        return
    lowered = user_input.lower()
    if lowered == "skip":
        return
    if lowered == "qa":
        question = input("Enter your question: ").strip()
        if not question:
            return
        qa = pipeline.answer_question(question, decision, topology=topology)
        artifacts.log_feedback(
            {
                "iter": decision.iter_index,
                "frame": decision.frame_index,
                "route": qa.route,
                "topology": qa.topology,
                "question": question,
                "answer": qa.answer,
                "latency_ms": qa.latency_ms,
                "answers": qa.answers,
                "safety_audit": qa.safety_audit,
            }
        )
        print(f"[{qa.topology.upper()}] {qa.answer}")
        return

    corrected = user_input.strip().upper()
    pipeline.apply_feedback(corrected, decision)
    artifacts.log_feedback(
        {
            "iter": decision.iter_index,
            "frame": decision.frame_index,
            "true_step": corrected,
            "pred": decision.step_prediction.fused_step,
            "chosen": decision.step_prediction.chosen,
            "w_s": pipeline.step_module.asf_head.g[0],
            "w_r": pipeline.step_module.asf_head.g[1],
        }
    )
    print(
        f"[HITL] Updated weights: "
        f"g[0]={pipeline.step_module.asf_head.g[0]:.3f}, "
        f"g[1]={pipeline.step_module.asf_head.g[1]:.3f}"
    )
