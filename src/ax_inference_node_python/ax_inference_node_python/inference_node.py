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

from axelera.runtime import Context, TensorInfo

#### Helper functions for preprocessing and postprocessing ####

def preprocess_frame(frame: np.ndarray, info: TensorInfo, mean: list[float], stddev: list[float]) -> tuple[np.ndarray, float, int, int]:
    """Preprocess frame by resizing with preserved aspect ratio and runtime padding"""
    batch, height, width, _ = info.unpadded_shape

    # Resize while preserving aspect ratio
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_w = int(frame.shape[1] * scale)
    resized_h = int(frame.shape[0] * scale)

    resized = cv2.resize(frame, (resized_w, resized_h))

    # Place resized into top-left corner of input (the rest will be padded by np.pad)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    x_offset = (width - resized_w) // 2
    y_offset = (height - resized_h) // 2

    image[y_offset:y_offset+resized_h, x_offset:x_offset+resized_w] = resized

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - np.array(mean)) / np.array(stddev)

    quantized = np.round(image / info.scale + info.zero_point).clip(-128, 127).astype(np.int8)

    padded = np.pad(quantized, info.padding[1:], mode="constant", constant_values=info.zero_point)

    if batch > 1:
        padded = np.repeat(padded[np.newaxis, ...], batch, axis=0)
    else:
        padded = padded[np.newaxis, ...]

    return padded, scale, x_offset, y_offset

def execute_onnx_postprocess(onnx_model_path: str, inputs_list: list[list[float]]) -> list[np.ndarray]:
    """
    Run the ONNX model with a list of lists as inputs, validate sizes, and return the output.
    """
    # Load the ONNX model with ONNX Runtime
    session = ort.InferenceSession(onnx_model_path)
    input_metadata = session.get_inputs()

    # Validate the number of inputs
    if len(inputs_list) != len(input_metadata):
        raise ValueError(f"Expected {len(input_metadata)} inputs, but got {len(inputs_list)}.")

    # Prepare the input feed dictionary
    input_feed = {
        input_meta.name: np.array(inputs_list[i], dtype=np.float32)
        for i, input_meta in enumerate(input_metadata)
    }

    # Run inference
    outputs = session.run(None, input_feed)
    return outputs

def extract_bounding_boxes(predictions: np.ndarray, has_objectness: bool, confidence_threshold: float) -> tuple[list[list[float]], list[float], list[int]]:
        """Parse predictions to extract bounding boxes, confidences, and class IDs."""
        boxes, confidences, class_ids = [], [], []

        for pred in predictions:
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:] if not has_objectness else pred[5:]
            objectness = pred[4] if has_objectness else 1.0

            class_id = np.argmax(class_scores)
            confidence = objectness * class_scores[class_id]

            if confidence > confidence_threshold:
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        return boxes, confidences, class_ids

def postprocess_model_output(onnx_model_path: str, inputs_list: list[list[float]], confidence_threshold: float, nms_threshold: float) -> tuple[str, list[tuple[int, float, list[float]]]]:
    """
    Postprocess ONNX model results to extract final detections.

    Args:
        onnx_results (list[np.ndarray]): List of ONNX model outputs.
        confidence_threshold (float): Minimum confidence threshold for filtering.
        nms_threshold (float): IoU threshold for Non-Maximum Suppression.

    Returns:
        tuple: (box_type, list[tuple]) where box_type indicates the format of the boxes (xyxy,xywh), 
               (class_id, confidence, (x1, y1, x2, y2)).
    """
    # Run ONNX postprocessing
    onnx_results = execute_onnx_postprocess(onnx_model_path, inputs_list)
    
    final_detections = []
    box_type = None

    for result in onnx_results:
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

        # ROS2 publishers and subscribers
        self.image_pub = self.create_publisher(Image, "/camera_frame_annotated", 10)
        self.subscription = self.create_subscription(Image, input_topic, self.image_callback, 10)
        self.publisher = self.create_publisher(String, output_topic, 10)

        self.timer = self.create_timer(0.05, self.process_worker_results)

        self.get_logger().info(f"Axelera inference node started with model: {self.model_name}")

    def load_labels(self, labels_path: Path) -> list[str]:
        """Load labels from a JSON file."""
        try:
            with labels_path.open('r') as f:
                data = json.load(f) 
                labels = data.get("labels", [])
                if not labels:
                    self.get_logger().warning(f"No labels found in {labels_path}")
                else:
                    self.get_logger().info(f"Loaded {len(labels)} labels from {labels_path}")                
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
            
            processed_input, self.scale, self.x_offset, self.y_offset = preprocess_frame(frame, self.input_infos[0], self.mean, self.stddev)
            
            self.get_logger().info(f"Frame {self.frame_id} preprocessed successfully.")
            self.get_logger().debug(f"Preprocessed input shape: {processed_input.shape}, scale: {self.scale}, x_offset: {self.x_offset}, y_offset: {self.y_offset}")

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

        # Perform ONNX model inference
        box_type, detections = postprocess_model_output(self.onnx_model_path, outs, confidence_threshold=self.conf_threshold, nms_threshold=self.nms_threshold)

        # Log the detections
        for class_id, confidence, (x1, y1, x2, y2) in detections:
            self.get_logger().debug(f"Class {self.labels[class_id]}, Confidence: {confidence:.2f}, Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        
        # Log number of detections
        self.get_logger().info(f"Number of detections: {len(detections)}")

        # Visualization and publishing
        if self.current_frame is not None:
            frame = self.plot_detections(detections, box_type)
            msg_out = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.image_pub.publish(msg_out)

        if detections:
            msg = String()
            msg.data = "\n".join(
                f"Detection: {label} ({confidence:.1f}%) at ({x1:.1f}, {y1:.1f}, {width:.1f}, {height:.1f})"
                for label, confidence, (x1, y1, width, height) in detections
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
