#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <iostream>
#include <typeinfo> 
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>

#include "axruntime/axruntime.hpp"
#include "opencv2/opencv.hpp"
#include <onnxruntime_cxx_api.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <optional>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect2f box;
};

// Model parameters
const std::array<float, 3> mean = { { 0.485, 0.456, 0.406 } };
const std::array<float, 3> stddev = { { 0.229, 0.224, 0.225 } };

using namespace std::string_literals;

constexpr auto DEFAULT_LABELS = "ax_datasets/labels/coco.names";
constexpr auto DEFAULT_MODEL_JSON = "../build/yolo11x-coco-onnx/yolo11x-coco-onnx/1/model.json";
constexpr auto DEFAULT_IMAGE = "../frame14.jpg";
constexpr auto DEFAULT_MODEL_ONNX = "../build/yolo11x-coco-onnx/yolo11x-coco-onnx/1/postprocess_graph.onnx";

#include <vector>
#include <stdexcept>
#include <vector>
#include <utility>
#include <cassert>

// Helper to compute unpadded shape
std::vector<size_t> compute_unpadded_shape(const size_t* dims,
      const size_t (*padding)[2],
      size_t ndim) {
    std::vector<size_t> unpadded;
    unpadded.reserve(ndim);

    for (size_t i = 0; i < ndim; ++i) {
    size_t pad_left  = padding[i][0];
    size_t pad_right = padding[i][1];
    assert(dims[i] >= (pad_left + pad_right));  // safety check
    unpadded.push_back(dims[i] - pad_left - pad_right);
    }

    return unpadded;
}

size_t get_flat_index_NHWC(size_t n, size_t h, size_t w, size_t c,
    size_t H, size_t W, size_t C) {
    return ((n * H + h) * W + w) * C + c;
    }

size_t get_flat_index_NCHW(size_t n, size_t c, size_t h, size_t w,
    size_t C, size_t H, size_t W) {
    return ((n * C + c) * H + h) * W + w;
    }

std::vector<float> transpose_NHWC_to_NCHW(const std::vector<float>& input,
                  size_t N, size_t H, size_t W, size_t C) {
    assert(input.size() == N * H * W * C && "Input size mismatch");

    std::vector<float> output(N * C * H * W);
    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
            for (size_t c = 0; c < C; ++c) {
                size_t in_idx = get_flat_index_NHWC(n, h, w, c, H, W, C);
                size_t out_idx = get_flat_index_NCHW(n, c, h, w, C, H, W);
                output[out_idx] = input[in_idx];
                }
            }
        }
    }
    return output;
}

std::vector<std::vector<float>> process_outputs(
    const std::vector<std::unique_ptr<std::int8_t[]>>& outputs,
    const std::vector<axrTensorInfo>& output_infos,
    rclcpp::Logger logger)
{
    std::vector<std::vector<float>> outs;

    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& info = output_infos[i];
        const int8_t* output = outputs[i].get();
        
        const auto& padded_shape = info.dims;  
        const auto& padding = info.padding;

        auto unpadded_shape = compute_unpadded_shape(padded_shape, padding, info.ndims);

        size_t N = unpadded_shape[0];
        size_t H = unpadded_shape[1];
        size_t W = unpadded_shape[2];
        size_t C = unpadded_shape[3];

        std::vector<float> sliced_output;
        sliced_output.reserve(N * H * W * C);

        // Extract values from unpadded region and dequantize
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    for (size_t c = 0; c < C; ++c) {
                        size_t n_in = n + padding[0][0];
                        size_t h_in = h + padding[1][0];
                        size_t w_in = w + padding[2][0];
                        size_t c_in = c + padding[3][0];

                        size_t input_idx = get_flat_index_NHWC(n_in, h_in, w_in, c_in,
                                                             padded_shape[1], padded_shape[2], padded_shape[3]);
                        float val = (output[input_idx] - info.zero_point) * info.scale;
                        sliced_output.push_back(val);
                    }
                }
            }
        }
        
        // Transpose NHWC -> NCHW
        auto transposed = transpose_NHWC_to_NCHW(sliced_output, N, H, W, C);
        outs.push_back(std::move(transposed));
    }
    return outs;
}

std::vector<Ort::Value> execute_onnx_postprocess(
    const std::string& onnx_model_path,
    const std::vector<std::vector<float>>& inputs_list,
    rclcpp::Logger logger)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_postprocess");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, onnx_model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_model_inputs = session.GetInputCount();

    if (inputs_list.size() != num_model_inputs) {
        throw std::invalid_argument("Number of inputs does not match model input count.");
    }

    // Prepare input names and tensors
    std::vector<const char*> input_names;
    std::vector<Ort::AllocatedStringPtr> owned_input_names;
    std::vector<Ort::Value> input_tensors;

    for (size_t i = 0; i < num_model_inputs; ++i) {
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
        input_names.push_back(input_name.get());
        owned_input_names.push_back(std::move(input_name));

        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = tensor_info.GetShape();

        size_t actual_size = inputs_list[i].size();
        for (auto& dim : input_shape) {
            if (dim == -1) {
                dim = static_cast<int64_t>(actual_size);
            }
        }

        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            allocator.GetInfo(),
            const_cast<float*>(inputs_list[i].data()),
            actual_size,
            input_shape.data(),
            input_shape.size()
        );

        input_tensors.emplace_back(std::move(tensor));
    }

    // Prepare output names
    size_t num_outputs = session.GetOutputCount();
    std::vector<const char*> output_names;
    std::vector<Ort::AllocatedStringPtr> owned_output_names;

    for (size_t i = 0; i < num_outputs; ++i) {
        Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
        output_names.push_back(output_name.get());
        owned_output_names.push_back(std::move(output_name));
    }

    // Run inference
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        input_names.size(),
        output_names.data(),
        output_names.size()
    );

    return output_tensors;
}

std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> 
extract_bounding_boxes(
    const std::vector<std::vector<float>>& predictions,
    bool has_objectness,
    float confidence_threshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (const auto& pred : predictions) {
        float conf;
        int cls_id;

        if (has_objectness) {
            float obj_conf = pred[4];  // objectness
            // Find class with highest confidence
            float max_cls_conf = 0.0f;
            int max_cls_id = -1;
            for (int c = 5; c < 85; ++c) {
                if (pred[c] > max_cls_conf) {
                    max_cls_conf = pred[c];
                    max_cls_id = c - 5;
                }
            }
            conf = obj_conf * max_cls_conf;
            cls_id = max_cls_id;
        } else {
            // Find class with highest class score
            float max_cls_conf = 0.0f;
            int max_cls_id = -1;
            for (int c = 4; c < 84; ++c) {
                if (pred[c] > max_cls_conf) {
                    max_cls_conf = pred[c];
                    max_cls_id = c - 4;
                }
            }
            conf = max_cls_conf;
            cls_id = max_cls_id;
        }

        if (conf < confidence_threshold)
            continue;

        // Convert from center+size format to corner coordinates
        float x_center = pred[0];
        float y_center = pred[1];
        float width = pred[2];
        float height = pred[3];
        float x1 = x_center - width / 2.0f;
        float y1 = y_center - height / 2.0f;
        float x2 = x_center + width / 2.0f;
        float y2 = y_center + height / 2.0f;

        // Fix vexing parse by using braces instead of parentheses
        cv::Rect box{cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2))};
        boxes.push_back(box);
        confidences.push_back(conf);
        class_ids.push_back(cls_id);
    }

    return { boxes, confidences, class_ids };
}

std::tuple<std::string, std::vector<Detection>>
postprocess_model_output(
    const std::string& onnx_model_path,
    const std::vector<std::vector<float>>& inputs_list,
    float confidence_threshold,
    float nms_threshold,
    rclcpp::Logger logger)
{
    // Run ONNX postprocess
    auto onnx_results = execute_onnx_postprocess(onnx_model_path, inputs_list, logger);

    std::vector<Detection> final_detections;
    std::string box_type;

    for (const auto& result : onnx_results) {
        bool has_objectness = false;
        std::vector<std::vector<float>> predictions;
        auto shape_info = result.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = shape_info.GetShape();

        const float* data = result.GetTensorData<float>();

        // Determine format and extract predictions
        if (shape.size() == 3 && shape[2] == 85) {
            // YOLOv5 format: shape = (1, N, 85)
            box_type = "xyxy";
            has_objectness = true;
            int64_t N = shape[1];

            for (int64_t i = 0; i < N; ++i) {
                std::vector<float> pred(85);
                for (int64_t j = 0; j < 85; ++j) {
                    pred[j] = data[i * 85 + j];
                }
                predictions.push_back(std::move(pred));
            }
        }
        else if (shape.size() == 3 && shape[1] == 84) {
            // YOLOv8 format: shape = (1, 84, N)
            box_type = "xyxy";
            has_objectness = false;
            int64_t N = shape[2];

            for (int64_t i = 0; i < N; ++i) {
                std::vector<float> pred(84);
                for (int64_t j = 0; j < 84; ++j) {
                    pred[j] = data[j * N + i]; 
                }
                predictions.push_back(std::move(pred));
            }
        }
        else {
            throw std::runtime_error("Unexpected result shape: cannot interpret predictions");
        }

        // Parse predictions -> boxes, confidences, class_ids
        auto [boxes, confidences, class_ids] = extract_bounding_boxes(predictions, has_objectness, confidence_threshold);

        // Apply Non-Maximum Suppression
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, indices);

        for (int idx : indices) {
            Detection det;
            det.class_id = class_ids[idx];
            det.confidence = confidences[idx];
            det.box = boxes[idx];
            final_detections.push_back(det);
        }
    }

    return { box_type, final_detections };
}

std::tuple<std::vector<std::int8_t>, float, int, int> preprocess_frame(
    cv::Mat frame, 
    const axrTensorInfo &info,
    const std::array<float, 3> &mean,
    const std::array<float, 3> &stddev)
{
    const auto height = info.dims[1];
    const auto width = info.dims[2];
    const auto channels = info.dims[3];
    const auto [y_pad_left, y_pad_right] = info.padding[1];
    const auto [x_pad_left, x_pad_right] = info.padding[2];
    const auto unpadded_height = height - y_pad_left - y_pad_right;
    const auto unpadded_width = width - x_pad_left - x_pad_right;

    // Calculate scale and offsets
    float scale = std::min(static_cast<float>(unpadded_width) / frame.cols,
                          static_cast<float>(unpadded_height) / frame.rows);
    int resized_width = static_cast<int>(frame.cols * scale);
    int resized_height = static_cast<int>(frame.rows * scale);

    int x_offset = (unpadded_width - resized_width) / 2;
    int y_offset = (unpadded_height - resized_height) / 2;

    // Resize the input image
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);

    // Create padded image
    cv::Mat padded_image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(padded_image(cv::Rect(x_offset, y_offset, resized_width, resized_height)));

    // Normalize and quantize
    std::vector<std::int8_t> quantized_image(height * width * channels, 
                                           static_cast<std::int8_t>(std::clamp(info.zero_point, -128, 127)));

    const std::array<float, 3> mul = {
        { 1.0f / stddev[0], 1.0f / stddev[1], 1.0f / stddev[2] }};
    const std::array<float, 3> add = {
        { -mean[0] / stddev[0], -mean[1] / stddev[1], -mean[2] / stddev[2] }};

    for (int y = 0; y < padded_image.rows; ++y) {
        for (int x = 0; x < padded_image.cols; ++x) {
            cv::Vec3b pixel = padded_image.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c) {
                float normalized_val = pixel[2-c] / 255.0f * mul[c] + add[c];
                float qf = normalized_val / info.scale + info.zero_point;
                float qf_clamped = std::clamp(qf, -128.0f, 127.0f);
                int8_t q = static_cast<std::int8_t>(std::round(qf_clamped));

                size_t idx = (y * width + x) * channels + c;
                if (idx < quantized_image.size()) {
                    quantized_image[idx] = q;
                }
            }
        }
    }

    return {quantized_image, scale, x_offset, y_offset};
}

cv::Mat plot_detections(
    const cv::Mat& frame,
    const std::vector<Detection>& detections,
    const std::string& box_type,
    const std::vector<std::string>& labels,
    const std::string& model_name,
    float scale,
    int x_offset,
    int y_offset)
{
    cv::Mat annotated_frame = frame.clone();

    for (const auto& detection : detections) {
        int class_id = detection.class_id;
        float confidence = detection.confidence;
        cv::Rect2f box = detection.box;

        std::string label = (class_id >= 0 && class_id < labels.size()) ? labels[class_id] : "Unknown";

        // Reverse scaling and padding transformations
        float x1 = (box.x - x_offset) / scale;
        float y1 = (box.y - y_offset) / scale;
        float x2 = ((box.x + box.width) - x_offset) / scale;
        float y2 = ((box.y + box.height) - y_offset) / scale;

        // Ensure integers and clamp to frame size
        int x1_int = std::max(0, std::min(static_cast<int>(std::round(x1)), frame.cols - 1));
        int y1_int = std::max(0, std::min(static_cast<int>(std::round(y1)), frame.rows - 1));
        int x2_int = std::max(0, std::min(static_cast<int>(std::round(x2)), frame.cols - 1));
        int y2_int = std::max(0, std::min(static_cast<int>(std::round(y2)), frame.rows - 1));

        // Draw bounding box
        cv::Scalar color = (confidence >= 0.5) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(annotated_frame, cv::Point(x1_int, y1_int), cv::Point(x2_int, y2_int), color, 2);

        // Add label and confidence
        std::ostringstream text;
        text << label << " " << static_cast<int>(confidence * 100) << "%";
        cv::putText(annotated_frame, text.str(), cv::Point(x1_int, std::max(0, y1_int - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);

        // Draw center point
        int cx = (x1_int + x2_int) / 2;
        int cy = (y1_int + y2_int) / 2;
        cv::circle(annotated_frame, cv::Point(cx, cy), 3, cv::Scalar(0, 0, 255), -1);
    }

    // Add model name
    cv::putText(annotated_frame, "Model used: " + model_name, cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    return annotated_frame;
}

std_msgs::msg::String create_detection_message(
    const std::vector<Detection>& detections, 
    const std::vector<std::string>& labels)
{
    std_msgs::msg::String msg;
    std::ostringstream ss;

    for (const auto& detection : detections) {
        int class_id = detection.class_id;
        float confidence = detection.confidence;
        cv::Rect2f box = detection.box;
        
        std::string label = (class_id >= 0 && class_id < labels.size()) ? labels[class_id] : "Unknown";

        ss << "Detection: " << label << " (" << confidence * 100 << "%) at ("
           << box.x << ", " << box.y << ", " << box.width << ", " << box.height << ")\n";
    }

    msg.data = ss.str();
    return msg;
}

auto read_labels(const std::string &path)
{
    std::vector<std::string> labels;
    std::ifstream file(path);
    for (std::string line; std::getline(file, line);) {
        labels.push_back(line);
    }
    return labels;
}

std::tuple<std::string, std::vector<std::string>, std::vector<std::string>>
parse_args(int argc, char **argv)
{
    std::string model_path = DEFAULT_MODEL_JSON;
    std::vector<std::string> labels;
    std::vector<std::string> images = {DEFAULT_IMAGE, DEFAULT_IMAGE};

    if (model_path.empty() || images.empty()) {
        std::cerr
            << "Usage: " << argv[0] << " [model.json] [labels.txt] [images-or-dirs...]\n"
            << "  model.json: path to the model file (default: " << DEFAULT_MODEL_JSON << ")\n"
            << "  labels.txt: path to the labels file (default: " << DEFAULT_LABELS << ")\n"
            << "  images-or-dirs: paths to images or directories containing images (default: " << DEFAULT_IMAGE << ")\n"
            << std::endl;
        std::exit(1);
    }

    if (labels.empty()) {
        const auto root = std::getenv("AXELERA_FRAMEWORK");
        labels = read_labels(root && root[0] != '\0' ? root + "/"s + DEFAULT_LABELS : DEFAULT_LABELS);
    }

    return {model_path, labels, images};
}

void logger(void *arg, axrLogLevel level, const char *msg)
{
    (void) arg;
    (void) level;
}

// Define a ROS 2 Node class
class AxeleraYoloInference : public rclcpp::Node
{
public:
    AxeleraYoloInference() : Node("axelera_yolo_inference")
    {
        RCLCPP_INFO(this->get_logger(), "Starting Axelera YOLO Inference Node...");

        // Declare parameters with default values
        this->declare_parameter("model_name", "yolo11x-coco-onnx");
        this->declare_parameter("aipu_cores", 4);
        this->declare_parameter("input_topic", "/camera_frame");
        this->declare_parameter("output_topic", "/detections_topic");
        this->declare_parameter("confidence_threshold", 0.25);
        this->declare_parameter("nms_threshold", 0.45);
        this->declare_parameter("mean", std::vector<double>{0.485, 0.456, 0.406});
        this->declare_parameter("stddev", std::vector<double>{0.229, 0.224, 0.225});

        // Get parameter values
        model_name_ = this->get_parameter("model_name").as_string();
        aipu_cores_ = this->get_parameter("aipu_cores").as_int();
        input_topic_ = this->get_parameter("input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
        nms_threshold_ = this->get_parameter("nms_threshold").as_double();
        
        auto mean_param = this->get_parameter("mean").as_double_array();
        auto stddev_param = this->get_parameter("stddev").as_double_array();
        
        // Convert to arrays
        for (size_t i = 0; i < 3 && i < mean_param.size(); ++i) {
            mean_[i] = static_cast<float>(mean_param[i]);
        }
        for (size_t i = 0; i < 3 && i < stddev_param.size(); ++i) {
            stddev_[i] = static_cast<float>(stddev_param[i]);
        }

        RCLCPP_INFO(this->get_logger(), "Parameters loaded:");
        RCLCPP_INFO(this->get_logger(), "  Model name: %s", model_name_.c_str());
        RCLCPP_INFO(this->get_logger(), "  AIPU cores: %d", aipu_cores_);
        RCLCPP_INFO(this->get_logger(), "  Input topic: %s", input_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Output topic: %s", output_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Confidence threshold: %.2f", confidence_threshold_);
        RCLCPP_INFO(this->get_logger(), "  NMS threshold: %.2f", nms_threshold_);

        // Load labels using default path
        const auto root = std::getenv("AXELERA_FRAMEWORK");
        this->labels_ = read_labels(root && root[0] != '\0' ? root + "/"s + DEFAULT_LABELS : DEFAULT_LABELS);

        // Initialize publishers and subscribers with parametric topic names
        std::string annotated_topic = input_topic_ + "_annotated";
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(annotated_topic, 10);
        detection_pub_ = this->create_publisher<std_msgs::msg::String>(output_topic_, 10);
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            input_topic_, 10, std::bind(&AxeleraYoloInference::image_callback, this, std::placeholders::_1));

        // Initialize runtime context
        ctx_ = axr::to_ptr(axr_create_context());
        if (!ctx_) {
            throw std::runtime_error("Failed to create runtime context");
        }
        axr_set_logger(ctx_.get(), AXR_LOG_WARNING, logger, nullptr);

        // Load model using parametric model name
        std::string model_json_path = "../build/" + model_name_ + "/" + model_name_ + "/1/model.json";
        model_ = axr_load_model(ctx_.get(), model_json_path.c_str());
        if (!model_) {
            throw std::runtime_error("Failed to load model from path: " + model_json_path);
        }

        // Get model information
        auto inputs = axr_num_model_inputs(model_);
        for (size_t n = 0; n != inputs; ++n) {
            input_infos_.push_back(axr_get_model_input(model_, n));
        }
        auto outputs = axr_num_model_outputs(model_);
        for (size_t n = 0; n != outputs; ++n) {
            output_infos_.push_back(axr_get_model_output(model_, n));
        }
        const auto batch_size = input_infos_[0].dims[0];

        // Connect to device
        connection_ = axr_device_connect(ctx_.get(), nullptr, batch_size, nullptr);
        if (!connection_) {
            throw std::runtime_error("Failed to connect to device");
        }

        // Create model instance with parametric aipu_cores
        const auto props = "input_dmabuf=0;num_sub_devices=" + std::to_string(batch_size)
                           + ";aipu_cores=" + std::to_string(aipu_cores_);
        auto properties = axr_create_properties(ctx_.get(), props.c_str());
        instance_ = axr_load_model_instance(connection_, model_, properties);
        if (!instance_) {
            throw std::runtime_error("Failed to create model instance");
        }

        // Prepare buffers
        input_args_.resize(inputs);
        output_args_.resize(outputs);
        input_data_.resize(inputs);
        output_data_.resize(outputs);

        for (int n = 0; n != inputs; ++n) {
            input_data_[n] = std::make_unique<std::int8_t[]>(axr_tensor_size(&input_infos_[n]));
            input_args_[n].ptr = input_data_[n].get();
            input_args_[n].fd = 0;
            input_args_[n].offset = 0;
        }
        for (int n = 0; n != outputs; ++n) {
            output_data_[n] = std::make_unique<std::int8_t[]>(axr_tensor_size(&output_infos_[n]));
            output_args_[n].ptr = output_data_[n].get();
            output_args_[n].fd = 0;
            output_args_[n].offset = 0;
        }

        RCLCPP_INFO(this->get_logger(), "Axelera YOLO Inference Node initialized successfully.");
    }

    ~AxeleraYoloInference()
    {
        instance_ = nullptr;
        connection_ = nullptr;
        model_ = nullptr;
        ctx_ = nullptr;
    }

private:
    std::vector<axrTensorInfo> input_infos_;
    std::vector<axrTensorInfo> output_infos_;
    std::vector<axrArgument> input_args_;
    std::vector<axrArgument> output_args_;
    std::vector<std::unique_ptr<std::int8_t[]>> input_data_;
    std::vector<std::unique_ptr<std::int8_t[]>> output_data_;
    std::shared_ptr<axrContext> ctx_;
    axrModel* model_ = nullptr;
    axrConnection* connection_ = nullptr;
    axrModelInstance* instance_ = nullptr;

    std::vector<std::string> labels_;  // Class variable for labels

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr detection_pub_;

    // Parametric member variables
    std::string model_name_;
    int aipu_cores_;
    std::string input_topic_;
    std::string output_topic_;
    float confidence_threshold_;
    float nms_threshold_;
    std::array<float, 3> mean_;
    std::array<float, 3> stddev_;

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) 
    {
        try {
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
            process_image(frame);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void process_image(const cv::Mat& frame) 
    {
        // Preprocess frame using parametric mean and stddev
        auto [processed_input, scale, x_offset, y_offset] = preprocess_frame(
            frame, input_infos_[0], mean_, stddev_);
        
        // Copy processed input to buffer
        std::memcpy(input_args_[0].ptr, processed_input.data(), processed_input.size());

        // Run model inference
        if (axr_run_model_instance(instance_, input_args_.data(), input_args_.size(),
                                   output_args_.data(), output_args_.size()) != AXR_SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Failed to run model instance");
            return;
        }

        // Process outputs
        auto dequantized_outputs = process_outputs(output_data_, output_infos_, this->get_logger());
        
        // Use parametric ONNX model path
        std::string onnx_model_path = "../build/" + model_name_ + "/" + model_name_ + "/1/postprocess_graph.onnx";
        
        // Use parametric confidence and NMS thresholds
        auto [box_type, detections] = postprocess_model_output(
            onnx_model_path, dequantized_outputs, confidence_threshold_, nms_threshold_, this->get_logger());
        
        RCLCPP_INFO(this->get_logger(), "Number of detections: %zu", detections.size());

        // Visualization and publishing using parametric model name
        cv::Mat annotated_frame = plot_detections(
            frame, detections, box_type, labels_, model_name_, scale, x_offset, y_offset);

        // Publish annotated image
        auto msg_out = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", annotated_frame).toImageMsg();
        msg_out->header.stamp = this->now();
        msg_out->header.frame_id = "camera_frame";
        image_pub_->publish(*msg_out);

        // Publish detections
        if (!detections.empty()) {
            auto detection_msg = create_detection_message(detections, labels_);
            detection_pub_->publish(detection_msg);
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AxeleraYoloInference>());
    rclcpp::shutdown();
    return 0;
}
