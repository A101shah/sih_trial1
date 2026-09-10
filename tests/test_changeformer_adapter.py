"""
Tests for ChangeDetectionModel adapter.
"""

import os
import sys
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from satquery.models.change_detection import ChangeDetectionModel


def test_adapter_initialization():
    print("=== Test 1: Adapter Initialization ===")
    model = ChangeDetectionModel()
    assert model.net_G is not None, "Model network failed to initialize"
    print(f"Model successfully initialized on device: {model.device}")
    print(f"Loaded checkpoint: {model.checkpoint_path}")
    return model


def test_custom_inference(model):
    print("\n=== Test 2: Custom Input Pair Inference ===")
    img_t1 = os.path.join("custom_input", "A", "image1.png")
    img_t2 = os.path.join("custom_input", "B", "image1.png")

    if not (os.path.isfile(img_t1) and os.path.isfile(img_t2)):
        print("custom_input test images not found, skipping pair")
        return

    result = model.predict(img_t1, img_t2)

    assert "change_mask" in result, "Result missing change_mask"
    assert "change_percentage" in result, "Result missing change_percentage"
    assert "confidence" in result, "Result missing confidence"
    assert "overlay_t1" in result, "Result missing overlay_t1"
    assert "overlay_t2" in result, "Result missing overlay_t2"
    assert "bboxes" in result, "Result missing bboxes"
    assert "metadata" in result, "Result missing metadata"

    print("Result keys verified:")
    print(f"  - Change percentage: {result['change_percentage']:.2f}%")
    print(f"  - Confidence: {result['confidence']:.4f}")
    print(f"  - Changed pixels: {result['changed_pixels']} / {result['total_pixels']}")
    print(f"  - Change mask shape: {result['change_mask'].shape}, dtype: {result['change_mask'].dtype}")
    print(f"  - Detected change clusters: {len(result['bboxes'])}")
    print(f"  - Metadata: {result['metadata']}")


def test_levir_sample_inference(model):
    print("\n=== Test 3: LEVIR Sample Inference ===")
    levir_a_dir = os.path.join("samples_LEVIR", "A")
    levir_b_dir = os.path.join("samples_LEVIR", "B")

    if not (os.path.isdir(levir_a_dir) and os.path.isdir(levir_b_dir)):
        print("samples_LEVIR directory not found, skipping")
        return

    sample_files = [f for f in os.listdir(levir_a_dir) if f.endswith(('.png', '.jpg'))]
    if not sample_files:
        print("No sample files found in samples_LEVIR/A")
        return

    sample_name = sample_files[0]
    t1_path = os.path.join(levir_a_dir, sample_name)
    t2_path = os.path.join(levir_b_dir, sample_name)

    print(f"Testing on LEVIR sample: {sample_name}")
    result = model.predict(t1_path, t2_path)
    print(f"  - Change percentage: {result['change_percentage']:.2f}%")
    print(f"  - Confidence: {result['confidence']:.4f}")
    print(f"  - Clusters detected: {len(result['bboxes'])}")
    assert isinstance(result["change_mask"], np.ndarray)


def test_in_memory_images(model):
    print("\n=== Test 4: In-Memory PIL & Numpy Arrays ===")
    arr_t1 = np.full((256, 256, 3), 100, dtype=np.uint8)
    arr_t2 = np.full((256, 256, 3), 100, dtype=np.uint8)
    # Add a white square in t2
    arr_t2[50:100, 50:100, :] = 255

    pil_t1 = Image.fromarray(arr_t1)
    pil_t2 = Image.fromarray(arr_t2)

    result = model.predict(pil_t1, pil_t2)
    print(f"  - Synthetic test change percentage: {result['change_percentage']:.2f}%")
    print(f"  - Confidence: {result['confidence']:.4f}")
    print(f"  - Clusters: {len(result['bboxes'])}")
    assert "change_mask" in result


if __name__ == "__main__":
    print("Starting SatQuery ChangeFormer Adapter Tests...\n")
    model = test_adapter_initialization()
    test_custom_inference(model)
    test_levir_sample_inference(model)
    test_in_memory_images(model)
    print("\n==========================================")
    print("ALL ADAPTER TESTS PASSED SUCCESSFULLY!")
    print("==========================================")
