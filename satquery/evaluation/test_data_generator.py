"""
Generates a small ground truth dataset for evaluating VQA and Grounding models using LEVIR-CD sample images.
"""
import os
import json
from PIL import Image

def generate_test_data():
    samples_dir = os.path.join("d:\\ChangeFormer\\samples_LEVIR", "A")
    output_dir = "d:\\ChangeFormer\\datasets\\test_suite"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(samples_dir):
        print(f"Skipping test data generation, {samples_dir} not found.")
        return
        
    images = sorted(os.listdir(samples_dir))[:5]
    if not images:
        return
        
    vqa_gt = []
    grounding_gt = []
    
    for i, img_name in enumerate(images):
        img_path = os.path.join(samples_dir, img_name)
        
        # Simple heuristics for creating some ground truth
        # LEVIR-CD usually contains residential buildings and roads
        vqa_gt.append({
            "image_path": img_path,
            "question": "Is there a building present?",
            "expected_answer": "yes"
        })
        vqa_gt.append({
            "image_path": img_path,
            "question": "Are there any airplanes?",
            "expected_answer": "no"
        })
        
        # Grounding
        # Since we don't have exact bboxes, we will just provide dummy bounding boxes for testing the pipeline
        grounding_gt.append({
            "image_path": img_path,
            "target_text": "building",
            "expected_bboxes": [[10, 10, 50, 50]] # Dummy box [ymin, xmin, ymax, xmax]
        })
        
    with open(os.path.join(output_dir, "vqa_gt.json"), "w") as f:
        json.dump(vqa_gt, f, indent=4)
        
    with open(os.path.join(output_dir, "grounding_gt.json"), "w") as f:
        json.dump(grounding_gt, f, indent=4)
        
    print(f"Generated test dataset with {len(vqa_gt)} VQA queries and {len(grounding_gt)} Grounding queries.")

if __name__ == "__main__":
    generate_test_data()
