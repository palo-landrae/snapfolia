import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
from app.schema import Classification, SpeedMetrics, ClassificationInferenceResult

class EfficientNetEngine:
    def __init__(self, engine_path: str, class_map: dict):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load the engine
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.class_map = class_map
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def preprocess(self, image_np: np.ndarray):
        # YOLO-style efficient resizing
        img = cv2.resize(image_np, (224, 224))
        img = img.astype(np.float32) / 255.0
        # Normalize (EfficientNet standard)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        return np.ascontiguousarray(img)

    def predict(self, image_np: np.ndarray, top_k: int = 3) -> ClassificationInferenceResult:
        start_total = time.perf_counter()

        # 1. Preprocess
        start_pre = time.perf_counter()
        processed_img = self.preprocess(image_np)
        np.copyto(self.inputs[0]['host'], processed_img.ravel())
        pre_ms = (time.perf_counter() - start_pre) * 1000

        # 2. Inference (Async)
        start_inf = time.perf_counter()
        # Transfer input data to GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        # Execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back to CPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        # Sync stream
        self.stream.synchronize()
        inf_ms = (time.perf_counter() - start_inf) * 1000

        # 3. Postprocess (Softmax & Top-K)
        start_post = time.perf_counter()
        output = self.outputs[0]['host']
        # Manual Softmax (since TRT usually outputs raw logits)
        exp_preds = np.exp(output - np.max(output))
        probs = exp_preds / exp_preds.sum()
        
        top_idxs = np.argsort(probs)[-top_k:][::-1]
        results = [
            Classification(
                class_id=int(i),
                class_name=self.class_map.get(int(i), "unknown"),
                confidence=float(probs[i])
            ) for i in top_idxs
        ]
        post_ms = (time.perf_counter() - start_post) * 1000
        total_ms = (time.perf_counter() - start_total) * 1000

        return ClassificationInferenceResult(
            status="success",
            results=results,
            message="TensorRT inference successful.",
            device="cuda (TensorRT)",
            speed_ms=SpeedMetrics(
                preprocess=pre_ms, inference=inf_ms, postprocess=post_ms, total=total_ms
            ),
        )