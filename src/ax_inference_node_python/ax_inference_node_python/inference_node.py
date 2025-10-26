#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import numpy as np
import cv2
import threading
import queue
from pathlib import Path
import json
import onnxruntime as ort
import time

from axelera.runtime import Context

#### Helper functions for preprocessing and postprocessing ####

def preprocess_frame(frame: np.ndarray, 
                     model_height: int, 
                     model_width: int, 
                     mean_array: np.ndarray, 
                     stddev_array: np.ndarray,
                     scale: float,
                     zero_point: int,
                     padding: tuple,
                     batch_size: int) -> tuple[np.ndarray, float, int, int]:
    """Preprocess frame by resizing with preserved aspect ratio and runtime padding"""
    
    # Resize while preserving aspect ratio
    scale_factor = min(model_width / frame.shape[1], model_height / frame.shape[0])
    resized_w = int(frame.shape[1] * scale_factor)
    resized_h = int(frame.shape[0] * scale_factor)

    resized = cv2.resize(frame, (resized_w, resized_h))

    # Place resized into center of input (the rest will be padded by np.pad)
    image = np.zeros((model_height, model_width, 3), dtype=np.uint8)
    x_offset = (model_width - resized_w) // 2
    y_offset = (model_height - resized_h) // 2

    image[y_offset:y_offset+resized_h, x_offset:x_offset+resized_w] = resized

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - mean_array) / stddev_array

    quantized = np.round(image / scale + zero_point).clip(-128, 127).astype(np.int8)

    padded = np.pad(quantized, padding, mode="constant", constant_values=zero_point)

    if batch_size > 1:
        padded = np.repeat(padded[np.newaxis, ...], batch_size, axis=0)
    else:
        padded = padded[np.newaxis, ...]

    return padded, scale_factor, x_offset, y_offset

def execute_onnx_postprocess(
    onnx_session: ort.InferenceSession,
    onnx_io_binding: ort.IOBinding,
    onnx_input_buffers: list[np.ndarray],
    inputs_list: list[np.ndarray]
) -> list[np.ndarray]:
    """
    Execute ONNX postprocessing using preallocated I/O binding and input buffers.   
    """
    if len(inputs_list) != len(onnx_input_buffers):
        raise ValueError(f"Expected {len(onnx_input_buffers)} inputs, but got {len(inputs_list)}.")

    # Copy new data into the preallocated buffers
    for i, arr in enumerate(inputs_list):
        np.copyto(onnx_input_buffers[i], arr, casting='same_kind')

    # Run with the provided binding (zero-copy execution)
    onnx_session.run_with_iobinding(onnx_io_binding)

    # Copy outputs to CPU (allocates NumPy arrays)
    onnx_outputs = onnx_io_binding.copy_outputs_to_cpu()

    return onnx_outputs


def extract_bounding_boxes(predictions: np.ndarray, has_objectness: bool, confidence_threshold: float):
    """
    Vectorized extraction of bounding boxes, confidences, and class IDs.
    Much faster than looping in Python.
    """
    t_start = time.perf_counter()
    
    if has_objectness:
        objectness = predictions[:, 4:5]  # shape (N,1)
        class_scores = predictions[:, 5:]  # shape (N, num_classes)
    else:
        objectness = np.ones((predictions.shape[0], 1), dtype=np.float32)
        class_scores = predictions[:, 4:]  # shape (N, num_classes)

    class_ids = np.argmax(class_scores, axis=1)
    confidences = (objectness[:, 0] * class_scores[np.arange(predictions.shape[0]), class_ids])

    # Apply confidence threshold
    mask = confidences > confidence_threshold
    if np.sum(mask) == 0:
        return [], [], []

    # Select only valid predictions
    preds = predictions[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    x_center = preds[:, 0]
    y_center = preds[:, 1]
    width = preds[:, 2]
    height = preds[:, 3]

    # Convert to xyxy
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    boxes = np.stack([x1, y1, x2, y2], axis=1).tolist()
    confidences = confidences.tolist()
    class_ids = class_ids.tolist()

    
    return boxes, confidences, class_ids

def postprocess_model_output(
    onnx_session: ort.InferenceSession,
    onnx_input_names: list[str],
    onnx_io_binding: ort.IOBinding,
    onnx_input_buffers: list[np.ndarray],
    inputs_list: list[np.ndarray],
    confidence_threshold: float,
    nms_threshold: float
) -> tuple[str, list[tuple[int, float, list[float]]]]:
    """
    Postprocess ONNX model results to extract final detections.

    Args:
        onnx_session: The ONNX inference session.
        onnx_input_names: List of input names for the ONNX model.
        onnx_io_binding: Pre-initialized ONNX I/O binding for zero-copy execution.
        onnx_input_buffers: Preallocated NumPy input buffers.
        inputs_list: List of input arrays for the ONNX model.
        confidence_threshold: Minimum confidence threshold for filtering.
        nms_threshold: IoU threshold for Non-Maximum Suppression.

    Returns:
        (box_type, detections) where detections = (class_id, confidence, (x1, y1, x2, y2))
    """
    # Run ONNX postprocessing using existing session
    onnx_results = execute_onnx_postprocess(onnx_session, onnx_io_binding, onnx_input_buffers, inputs_list)
    
    final_detections = []
    box_type = None

    for i, result in enumerate(onnx_results):
        if result.ndim == 3 and result.shape[2] == 85:  # YOLOv5 format
            box_type = 'xyxy'
            predictions = result[0]  # shape: (N, 85)
            has_objectness = True
        elif result.ndim == 3 and result.shape[1] == 84:  # YOLOv8 format
            box_type = 'xyxy'
            predictions = result[0].T  # shape: (N, 84)
            has_objectness = False
        else:
            raise ValueError(f"Unexpected result shape: {result.shape}. Expected (1, N, 85) or (1, 84, N).")

        # Parse predictions
        boxes, confidences, class_ids = extract_bounding_boxes(predictions, has_objectness, confidence_threshold)
        
        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        if len(indices) > 0:
            for i in indices.flatten():
                final_detections.append((class_ids[i], confidences[i], boxes[i]))

    return box_type, final_detections

### Worker thread and ROS2 Node ###

# Worker thread
class CameraWorker(threading.Thread):
    def __init__(self, instance):
        super().__init__()
        self.instance = instance
        self.inqueue = queue.Queue(maxsize=2)
        self.outqueue = queue.Queue()
        self.running = True
        self.start()

    def run(self):
        while self.running:
            try:
                x = self.inqueue.get(timeout=0.1)
                if x is None:
                    break
                frame_id, *io = x
                try:                   
                    self.instance.run(*io)
                    
                except Exception as e:
                    self.outqueue.put(e)
                    break
                # Pass timing along with the results
                self.outqueue.put((frame_id, io[1]))
            except queue.Empty:
                continue

    def push(self, frame_id, inputs, outputs):
        try:
            self.inqueue.put([frame_id, inputs, outputs], block=False)
            return True
        except queue.Full:
            return False

    def pop(self):
        try:
            x = self.outqueue.get(block=False)
            if isinstance(x, Exception):
                raise x
            return x
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.inqueue.put(None)

# ROS2 Node
class AxeleraYoloInference(Node):
    def __init__(self):
        super().__init__('axelera_camera_classifier')
        
        self.current_frame = None
        self.frame_id = 0
        
        # Declare ROS parameters
        self.declare_parameter('model_name', '')
        self.declare_parameter('aipu_cores', 4)
        self.declare_parameter('input_topic', '/camera_frame')
        self.declare_parameter('output_topic', '/detections_topic')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('mean', [0.485, 0.456, 0.406])  
        self.declare_parameter('stddev', [0.229, 0.224, 0.225])    

        # Load parameters
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value  
        aipu_cores = self.get_parameter('aipu_cores').get_parameter_value().integer_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.nms_threshold = self.get_parameter('nms_threshold').get_parameter_value().double_value
        self.mean = self.get_parameter('mean').get_parameter_value().double_array_value  
        self.stddev = self.get_parameter('stddev').get_parameter_value().double_array_value  
        
        model_path = Path(f'../build/{self.model_name}/{self.model_name}/1/model.json')
        labels_path = Path(f'../build/{self.model_name}/{self.model_name}/model_info.json')
        self.onnx_model_path = Path(f'../build/{self.model_name}/{self.model_name}/1/postprocess_graph.onnx')
                    
        if model_path.is_dir():
            model_path = model_path / "model.json"

        # Load labels dynamically from the JSON file
        self.labels = self.load_labels(Path(labels_path))

        # Load model and runtime
        self.ctx = Context()
        self.model = self.ctx.load_model(model_path)

        self.input_infos = self.model.inputs()
        self.output_infos = self.model.outputs()

        self.batch_size = self.input_infos[0].shape[0]
        self.inputs = [np.zeros(t.shape, dtype=np.int8) for t in self.input_infos]
        self.outputs = [np.zeros(t.shape, dtype=np.int8) for t in self.output_infos]

        connection = self.ctx.device_connect(None, self.batch_size)
        self.instance = connection.load_model_instance(
            self.model,
            num_sub_devices=self.batch_size,
            aipu_cores=aipu_cores
        )

        self.worker = CameraWorker(self.instance)
        self.bridge = CvBridge()

        # Load ONNX model session
        self.onnx_session = ort.InferenceSession(str(self.onnx_model_path))
        self.onnx_input_names = [input_meta.name for input_meta in self.onnx_session.get_inputs()]
        
        # --- PRECOMPUTED PREPROCESSING VALUES ---
        # Extract and precompute values that don't change between frames
        input_info = self.input_infos[0]
        self.model_height = input_info.unpadded_shape[1]  # Height from tensor info
        self.model_width = input_info.unpadded_shape[2]   # Width from tensor info
        self.mean_array = np.array(self.mean, dtype=np.float32)
        self.stddev_array = np.array(self.stddev, dtype=np.float32)
        self.tensor_scale = input_info.scale
        self.tensor_zero_point = input_info.zero_point
        self.tensor_padding = input_info.padding[1:]  # Skip batch dimension padding
        
        # --- Optimized ONNX setup ---
        self.onnx_input_metas = self.onnx_session.get_inputs()
        self.onnx_output_metas = self.onnx_session.get_outputs()

        # Precompute input shapes (replace -1 with 1)
        self.onnx_input_shapes = [
            tuple(1 if dim == -1 or dim is None else dim for dim in meta.shape)
            for meta in self.onnx_input_metas
        ]

        # Preallocate CPU input buffers
        self.onnx_input_buffers = [
            np.empty(shape, dtype=np.float32)
            for shape in self.onnx_input_shapes
        ]

        # Preallocate I/O binding
        self.onnx_io_binding = self.onnx_session.io_binding()

        # Bind all inputs (CPU binding; zero-copy)
        for name, arr in zip(self.onnx_input_names, self.onnx_input_buffers):
            self.onnx_io_binding.bind_input(
                name=name,
                device_type='cpu',
                device_id=0,
                element_type=np.float32,
                shape=arr.shape,
                buffer_ptr=arr.__array_interface__['data'][0]
            )

        # Bind all outputs to CPU (ONNX allocates them automatically)
        for output_meta in self.onnx_output_metas:
            self.onnx_io_binding.bind_output(output_meta.name, 'cpu', 0)


        # ROS2 publishers and subscribers
        self.image_pub = self.create_publisher(Image, "/camera_frame_annotated", 10)
        self.subscription = self.create_subscription(Image, input_topic, self.image_callback, 10)
        self.publisher = self.create_publisher(String, output_topic, 10)

        self.timer = self.create_timer(0.016, self.process_worker_results)

        self.get_logger().info(f"Axelera inference node started with model: {self.model_name}")

    def load_labels(self, labels_path: Path) -> list[str]:
        """Load labels from a JSON file."""
        try:
            with labels_path.open('r') as f:
                data = json.load(f) 
                labels = data.get("labels", [])
                if not labels:
                    self.get_logger().warning(f"No labels found in {labels_path}")               
                return labels
        except FileNotFoundError:
            self.get_logger().error(f"Labels file not found: {labels_path}")
            return []
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Error parsing JSON in {labels_path}: {e}")
            return []
        except Exception as e:
            self.get_logger().error(f"Unexpected error loading labels: {e}")
            return []    

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_frame = frame.copy()
            
            processed_input, self.scale, self.x_offset, self.y_offset = preprocess_frame(
                frame, 
                self.model_height, 
                self.model_width, 
                self.mean_array, 
                self.stddev_array, 
                self.tensor_scale, 
                self.tensor_zero_point, 
                self.tensor_padding, 
                self.batch_size
            )
            
            self.inputs[0][:] = processed_input
            
            pushed = self.worker.push(self.frame_id, self.inputs, self.outputs)
            
            if pushed:
                self.frame_id += 1
            
        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")
    
    def plot_detections(self, detections: list[tuple[int, float, list[float]]], box_type: str) -> np.ndarray:
        """
        Plot detections on the original frame with scaling and padding reversal.
        """
        frame = self.current_frame.copy()

        for detection in detections:
            class_id, confidence, box = detection
            label = self.labels[class_id]

            if len(box) != 4:
                raise ValueError("Bounding box should have 4 values")

            if box_type == 'xywh':
                x, y, w, h = box
                x1 = float(x)
                y1 = float(y)
                x2 = x1 + float(w)
                y2 = y1 + float(h)
            elif box_type == 'xyxy':
                x1, y1, x2, y2 = map(float, box)
            else:
                raise ValueError(f"Unsupported box_format '{box_type}', use 'xyxy' or 'xywh'.")

            # Map from model coordinates (640x640) back to original frame space
            x1 = (x1 - self.x_offset) / self.scale
            y1 = (y1 - self.y_offset) / self.scale
            x2 = (x2 - self.x_offset) / self.scale
            y2 = (y2 - self.y_offset) / self.scale

            # Ensure integers
            x1, y1, x2, y2 = map(lambda v: int(round(v)), [x1, y1, x2, y2])

            # Clamp to frame size to avoid drawing outside image
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, min(x1, w_frame - 1))
            y1 = max(0, min(y1, h_frame - 1))
            x2 = max(0, min(x2, w_frame - 1))
            y2 = max(0, min(y2, h_frame - 1))

            # Draw center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            color = (0, 255, 0) if confidence >= 0.5 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {confidence:.0%}"
            cv2.putText(frame, text, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), radius=3, color=(0, 0, 255), thickness=-1)

        # Change the font color for the model name to black
        cv2.putText(frame, f"Model used: {self.model_name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return frame
    
    def process_worker_results(self):
        """Check for worker output and process detections."""
        result = self.worker.pop()         
        if result is None:
            self.get_logger().debug("No output from worker yet.")
            return
        
        _, outputs = result
        outs = []

        for i, output in enumerate(outputs):
            slices = tuple(slice(b, -e if e else None) for b, e in self.output_infos[i].padding)
            sliced_output = output[slices]
            sliced_output = (sliced_output.astype(np.float32) - self.output_infos[i].zero_point) * self.output_infos[i].scale
            sliced_output = sliced_output.transpose(0, 3, 1, 2)
            outs.append(sliced_output)
            
        # Perform ONNX model inference using existing session
        box_type, detections = postprocess_model_output(self.onnx_session, self.onnx_input_names,self.onnx_io_binding, self.onnx_input_buffers,  outs, confidence_threshold=self.conf_threshold, nms_threshold=self.nms_threshold)

        # Log the detections
        for class_id, confidence, (x1, y1, x2, y2) in detections:
            self.get_logger().debug(f"Class {self.labels[class_id]}, Confidence: {confidence:.2f}, Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

        if self.current_frame is not None:
            frame = self.plot_detections(detections, box_type)
            msg_out = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.image_pub.publish(msg_out)

        if detections:
            msg = String()
            msg.data = "\n".join(
                f"Detection: {self.labels[class_id]} ({confidence:.1f}%) at ({x1:.1f}, {y1:.1f}, {width:.1f}, {height:.1f})"
                for class_id, confidence, (x1, y1, width, height) in detections
            )
            self.publisher.publish(msg)

    def destroy_node(self):       
        self.worker.stop()
        self.worker.join()
        self.ctx.__exit__(None, None, None)
        super().destroy_node()

# Main entry point
def main(args=None):
    # run_prebuild()
    rclpy.init(args=args)
    node = AxeleraYoloInference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down inference node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()
