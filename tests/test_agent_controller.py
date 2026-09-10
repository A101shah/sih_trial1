"""
Technically honest unit and integration tests for SatQuery AI Agent Controller.
Verifies real ChangeFormer execution and confirms that unconfigured specialists
explicitly report NOT_CONFIGURED with no fabricated outputs.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from satquery.agent.controller import AgentController
from satquery.reporting.export import ReportExporter


def run_all_tests():
    print("==========================================================")
    print("SatQuery AI: Honest Verification & Specialist Status Tests")
    print("==========================================================")

    print("\n[Phase A] Initializing AgentController...")
    controller = AgentController()
    print("AgentController initialized successfully.")

    # 1. Bi-Temporal Change Detection with ChangeFormer (REAL MODEL)
    print("\n[Test 1] Bi-temporal Change Detection (Genuine ChangeFormerV6)...")
    t1_path = os.path.join("samples_LEVIR", "A", "test_102_0512_0000.png")
    t2_path = os.path.join("samples_LEVIR", "B", "test_102_0512_0000.png")

    if os.path.isfile(t1_path) and os.path.isfile(t2_path):
        res_cd = controller.execute(
            image_1=t1_path,
            image_2=t2_path,
            question="What changed between these two dates?",
            task="change_analysis"
        )
        print(f"  -> Task: {res_cd['task']}")
        print(f"  -> Detected change: {res_cd['change_percentage']:.2f}%")
        print(f"  -> Answer: {res_cd['answer']}")
        print(f"  -> Model Confidence: {res_cd['confidence']:.4f} ({res_cd['confidence_source']})")
        print(f"  -> Specialist Statuses: {res_cd['specialist_statuses']}")
        print(f"  -> Trace steps: {res_cd['trace']}")

        assert res_cd["task"] == "change_analysis"
        assert res_cd["specialist_statuses"]["ChangeFormerV6"] == "EXECUTED"
        assert res_cd["specialist_statuses"]["CDVQA"] == "NOT_CONFIGURED"
        assert isinstance(res_cd["change_mask"], np.ndarray)
        assert res_cd["change_percentage"] > 0
        assert res_cd["confidence"] is not None and res_cd["confidence"] > 0.90
        print("  [PASS] ChangeFormer executed genuinely with real logits and spatial outputs.")
    else:
        print("  Skipping LEVIR test (file not found).")

    synthetic_optical = np.full((256, 256, 3), 120, dtype=np.uint8)

    # 2. Single-Image VQA (UNCONFIGURED)
    print("\n[Test 2] Single-Image VQA (Checking NOT_CONFIGURED status)...")
    res_vqa = controller.execute(
        image_1=synthetic_optical,
        question="What type of land cover is visible?",
        task="single_vqa"
    )
    print(f"  -> Task: {res_vqa['task']}")
    print(f"  -> Answer: {res_vqa['answer']}")
    print(f"  -> Confidence: {res_vqa['confidence']}")
    print(f"  -> Specialist Statuses: {res_vqa['specialist_statuses']}")
    assert res_vqa["task"] == "single_vqa"
    assert res_vqa["specialist_statuses"]["RSVLM"] == "NOT_CONFIGURED"
    assert res_vqa["confidence"] is None
    assert "not configured" in res_vqa["answer"].lower()
    print("  [PASS] Single-Image VQA honestly reported NOT_CONFIGURED with no fake answer.")

    # 3. Scene Captioning (UNCONFIGURED)
    print("\n[Test 3] Scene Captioning (Checking NOT_CONFIGURED status)...")
    res_cap = controller.execute(
        image_1=synthetic_optical,
        task="captioning"
    )
    print(f"  -> Task: {res_cap['task']}")
    print(f"  -> Answer: {res_cap['answer']}")
    print(f"  -> Confidence: {res_cap['confidence']}")
    assert res_cap["task"] == "captioning"
    assert res_cap["specialist_statuses"]["RSVLM"] == "NOT_CONFIGURED"
    assert res_cap["confidence"] is None
    assert "not configured" in res_cap["answer"].lower()
    print("  [PASS] Scene captioning honestly reported NOT_CONFIGURED with no fake caption.")

    # 4. Text-Guided Grounding (UNCONFIGURED)
    print("\n[Test 4] Text-Guided Grounding (Checking NOT_CONFIGURED status)...")
    res_ground = controller.execute(
        image_1=synthetic_optical,
        question="Where is the built-up region?",
        task="grounding"
    )
    print(f"  -> Task: {res_ground['task']}")
    print(f"  -> Grounded regions: {len(res_ground['bboxes'])}")
    print(f"  -> Answer: {res_ground['answer']}")
    print(f"  -> Confidence: {res_ground['confidence']}")
    assert res_ground["task"] == "grounding"
    assert res_ground["specialist_statuses"]["RSGroundingEngine"] == "NOT_CONFIGURED"
    assert res_ground["bboxes"] == []
    assert res_ground["confidence"] is None
    assert "not configured" in res_ground["answer"].lower()
    print("  [PASS] Grounding honestly reported NOT_CONFIGURED with 0 fake bounding boxes.")

    # 5. Optical + SAR Cross-Modal Reasoning (UNCONFIGURED REASONING)
    print("\n[Test 5] Optical + SAR (Checking Preprocessing vs Reasoning status)...")
    synthetic_sar = np.full((256, 256), 40, dtype=np.uint8)
    res_fusion = controller.execute(
        image_1=synthetic_optical,
        image_2=synthetic_sar,
        question="Analyze optical vs SAR",
        task="optical_sar_fusion",
        modality_1="optical",
        modality_2="sar"
    )
    print(f"  -> Task: {res_fusion['task']}")
    print(f"  -> Answer: {res_fusion['answer']}")
    print(f"  -> Confidence: {res_fusion['confidence']}")
    print(f"  -> Specialist Statuses: {res_fusion['specialist_statuses']}")
    assert res_fusion["task"] == "optical_sar_fusion"
    assert res_fusion["specialist_statuses"]["OpticalPreprocessor"] == "EXECUTED"
    assert res_fusion["specialist_statuses"]["SARPreprocessor"] == "EXECUTED"
    assert res_fusion["specialist_statuses"]["OpticalSARFusionModel"] == "NOT_CONFIGURED"
    assert res_fusion["confidence"] is None
    assert "not configured" in res_fusion["answer"].lower()
    print("  [PASS] Optical+SAR preprocessing executed genuinely while reasoning honestly reported NOT_CONFIGURED.")

    # 6. Report Generation
    print("\n[Test 6] Report Exporter...")
    md_report = ReportExporter.to_markdown(res_cd)
    assert "[EXECUTED]" in md_report
    assert "[NOT CONFIGURED]" in md_report
    print("  [PASS] Report cleanly distinguishes EXECUTED vs NOT CONFIGURED specialists.")

    print("\n==========================================================")
    print("ALL HONESTY VERIFICATION TESTS PASSED (100%)")
    print("==========================================================")


if __name__ == "__main__":
    run_all_tests()
