import torch

import numpy as np
import time
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from app.schema import Classification, SpeedMetrics, ClassificationInferenceResult
from typing import cast


class EfficientNetEngine:
    def __init__(self, model_path: str, class_map: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Rebuild EfficientNet
        self.model = efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(1280, len(class_map))

        # Extract weights
        state_dict = checkpoint["model_state_dict"]

        # Remove "module." prefix if trained with DataParallel
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Load weights
        self.model.load_state_dict(state_dict)

        # Move to device
        self.model.to(self.device)
        self.model.eval()

        self.class_map = class_map
        # Define the transform
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),  # Converts np.ndarray (H,W,C) to PIL
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(
        self, image_np: np.ndarray, top_k: int = 3
    ) -> ClassificationInferenceResult:
        """
        image_np: Expected shape (H, W, 3) in RGB format.
        If using OpenCV (BGR), call image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) first.
        """
        start_total = time.perf_counter()

        # 1. Preprocess
        start_pre = time.perf_counter()
        # Move NumPy array through transforms to get a GPU-ready tensor
        transformed_img = cast(torch.Tensor, self.transform(image_np))
        img_tensor = transformed_img.unsqueeze(0).to(self.device)
        pre_ms = (time.perf_counter() - start_pre) * 1000

        # 2. Inference
        start_inf = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
        inf_ms = (time.perf_counter() - start_inf) * 1000

        # 3. Postprocess
        start_post = time.perf_counter()
        top_probs, top_idxs = probs.topk(top_k)
        results = []
        for prob, idx in zip(top_probs[0], top_idxs[0]):
            class_id = int(idx.item())
            results.append(
                Classification(
                    class_id=class_id,
                    class_name=self.class_map.get(class_id, "unknown"),
                    confidence=float(prob.item()),
                )
            )

        post_ms = (time.perf_counter() - start_post) * 1000
        total_ms = (time.perf_counter() - start_total) * 1000

        return ClassificationInferenceResult(
            status="success",
            results=results,
            message=f"Inference completed successfully. Top-{top_k} classifications extracted.",
            device=str(self.device),
            speed_ms=SpeedMetrics(
                preprocess=pre_ms, inference=inf_ms, postprocess=post_ms, total=total_ms
            ),
        )
