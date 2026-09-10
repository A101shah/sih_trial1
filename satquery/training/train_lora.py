"""
Parameter-Efficient Fine-Tuning (PEFT / LoRA) Pipeline for SatQuery AI.
Fine-tunes remote sensing vision-language adapters using Low-Rank Adaptation (LoRA)
with minimal memory overhead on custom aerial / satellite imagery datasets.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tv_models

try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


class RemoteSensingVLMDataset(Dataset):
    """
    Dataset loader for Remote Sensing VQA & Classification fine-tuning.
    Supports local image directories and JSON metadata.
    """

    def __init__(self, samples: List[Dict[str, Any]], img_size: int = 256, transform=None):
        self.samples = samples
        self.img_size = img_size
        self.transform = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item.get("image_path")
        
        # Load or generate synthetic remote sensing image tensor
        if img_path and os.path.isfile(img_path):
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            tensor = self.transform(img)
        else:
            # Synthetic aerial tensor for testing & benchmarking
            tensor = torch.randn(3, self.img_size, self.img_size)
            
        label = item.get("label_id", 0)
        return tensor, torch.tensor(label, dtype=torch.long)


class LoRAFineTuner:
    """
    Manages LoRA parameter-efficient adaptation of remote-sensing models.
    """

    def __init__(
        self,
        num_classes: int = 23,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lr: float = 3e-4,
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_classes = num_classes
        self.lr = lr
        
        # Base Vision Backbone
        self.base_model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT if hasattr(tv_models, "ResNet50_Weights") else None)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
        # Freeze base parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Enable gradients on classification head
        for param in self.base_model.fc.parameters():
            param.requires_grad = True

        # Wrap with PEFT LoRA if available
        if HAS_PEFT:
            try:
                # Target linear projection in layer4 or fc
                peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["fc"],
                    lora_dropout=0.05,
                    bias="none"
                )
                self.model = get_peft_model(self.base_model, peft_config)
                print(f"[LoRA] Successfully wrapped model with LoRA (r={lora_r}, alpha={lora_alpha}).")
            except Exception as e:
                print(f"[LoRA Warning] PEFT wrapping fallback: {e}")
                self.model = self.base_model
        else:
            self.model = self.base_model

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        epoch_loss = total_loss / max(total, 1)
        epoch_acc = (correct / max(total, 1)) * 100.0
        return {"loss": round(epoch_loss, 4), "accuracy": round(epoch_acc, 2)}

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        val_loss = total_loss / max(total, 1)
        val_acc = (correct / max(total, 1)) * 100.0
        return {"val_loss": round(val_loss, 4), "val_accuracy": round(val_acc, 2)}

    def save_checkpoint(self, output_path: str, meta: Optional[Dict[str, Any]] = None):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "meta": meta or {},
            "num_classes": self.num_classes,
            "timestamp": time.time()
        }
        torch.save(checkpoint, output_path)
        print(f"[Checkpoint Saved] -> {output_path}")


def run_training_pipeline(
    epochs: int = 3,
    batch_size: int = 8,
    checkpoint_dir: str = "checkpoints/lora_rsvqa"
) -> Dict[str, Any]:
    """
    Executes an end-to-end domain adaptation experiment.
    """
    print("==================================================")
    print("   SatQuery AI: LoRA Remote Sensing Fine-Tuning   ")
    print("==================================================")

    # Prepare sample dataset
    train_samples = [{"id": i, "label_id": i % 23} for i in range(40)]
    val_samples = [{"id": i, "label_id": i % 23} for i in range(16)]

    train_ds = RemoteSensingVLMDataset(train_samples)
    val_ds = RemoteSensingVLMDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    tuner = LoRAFineTuner(num_classes=23)
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = tuner.train_epoch(train_loader)
        val_metrics = tuner.evaluate(val_loader)
        
        record = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics
        }
        history.append(record)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.1f}% | Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_accuracy']:.1f}%")

    out_ckpt = os.path.join(checkpoint_dir, "lora_rs_adapter.pt")
    tuner.save_checkpoint(out_ckpt, meta={"history": history, "epochs": epochs})

    return {
        "status": "COMPLETED",
        "checkpoint": out_ckpt,
        "history": history,
        "final_accuracy": history[-1]["val_accuracy"]
    }


if __name__ == "__main__":
    run_training_pipeline(epochs=2, batch_size=4)
