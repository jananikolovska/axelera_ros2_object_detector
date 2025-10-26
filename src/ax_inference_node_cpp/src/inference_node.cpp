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
#include <opencv2/core.hpp>

#include "axruntime/axruntime.hpp"
#include "opencv2/opencv.hpp"
#include <onnxruntime_cxx_api.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <optional>

#include <tuple>
#include <algorithm>
#include <utility>
#include <cassert>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect2f box;
};

// Model parameters
constexpr auto DEFAULT_LABELS = "ax_datasets/labels/coco.names";

using namespace std::string_literals;

// Helper to compute unpadded shape
std::vector<size_t> compute_unpadded_shape(const size_t* dims,
                                           const size_t (*padding)[2],
                                           size_t ndim) {
    std::vector<size_t> unpadded;
    unpadded.reserve(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        size_t pad_left = padding[i][0];
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
    rclcpp::Logger logger) {
    std::vector<std::vector<float>> outs;
    outs.reserve(outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& info = output_infos[i];
        const int8_t* output = outputs[i].get();

        const auto& padded_shape = info.dims;
        const auto& padding = info.padding;

        auto unpadded_shape = compute_unpadded_shape(padded_shape, padding, info.ndims);

        const size_t N = unpadded_shape[0];
        const size_t H = unpadded_shape[1];
        const size_t W = unpadded_shape[2];
        const size_t C = unpadded_shape[3];

        // Precompute all strides and constants
        const size_t padded_H = padded_shape[1];
        const size_t padded_W = padded_shape[2];
        const size_t padded_C = padded_shape[3];

        const size_t input_stride_n = padded_H * padded_W * padded_C;
        const size_t input_stride_h = padded_W * padded_C;
        const size_t input_stride_w = padded_C;

        const size_t output_stride_n = C * H * W;
        const size_t output_stride_c = H * W;
        const size_t output_stride_h = W;

        const size_t pad_n = padding[0][0];
        const size_t pad_h = padding[1][0];
        const size_t pad_w = padding[2][0];
        const size_t pad_c = padding[3][0];

        // Precompute quantization parameters
        const float scale = info.scale;
        const float zero_point_f = static_cast<float>(info.zero_point);

        std::vector<float> result(N * C * H * W);
        float* result_ptr = result.data();

        // Optimized nested loops with better memory access patterns
        for (size_t n = 0; n < N; ++n) {
            const size_t n_in_base = (n + pad_n) * input_stride_n;
            const size_t n_out_base = n * output_stride_n;

            for (size_t h = 0; h < H; ++h) {
                const size_t h_in_base = n_in_base + (h + pad_h) * input_stride_h;

                for (size_t w = 0; w < W; ++w) {
                    const size_t w_in_base = h_in_base + (w + pad_w) * input_stride_w;
                    const size_t w_out_base = n_out_base + h * output_stride_h + w;

                    // Process multiple channels at once for better cache utilization
                    const int8_t* input_row = output + w_in_base + pad_c;

                    // Vectorized processing of channels (process 4 at a time when possible)
                    size_t c = 0;
                    for (; c + 3 < C; c += 4) {
                        // Process 4 channels simultaneously
                        const int8_t v0 = input_row[c];
                        const int8_t v1 = input_row[c + 1];
                        const int8_t v2 = input_row[c + 2];
                        const int8_t v3 = input_row[c + 3];

                        result_ptr[w_out_base + c * output_stride_c] = (v0 - zero_point_f) * scale;
                        result_ptr[w_out_base + (c + 1) * output_stride_c] = (v1 - zero_point_f) * scale;
                        result_ptr[w_out_base + (c + 2) * output_stride_c] = (v2 - zero_point_f) * scale;
                        result_ptr[w_out_base + (c + 3) * output_stride_c] = (v3 - zero_point_f) * scale;
                    }

                    // Handle remaining channels
                    for (; c < C; ++c) {
                        const int8_t value = input_row[c];
                        result_ptr[w_out_base + c * output_stride_c] = (value - zero_point_f) * scale;
                    }
                }
            }
        }

        outs.push_back(std::move(result));
    }

    return outs;
}

std::vector<Ort::Value> execute_onnx_postprocess(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const std::vector<const char*>& input_names,
    const std::vector<const char*>& output_names,
    const std::vector<std::vector<float>>& inputs_list) {
    std::vector<Ort::Value> input_tensors;
    size_t num_model_inputs = session.GetInputCount();

    if (inputs_list.size() != num_model_inputs)
        throw std::invalid_argument("Number of inputs does not match model input count.");

    for (size_t i = 0; i < num_model_inputs; ++i) {
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = tensor_info.GetShape();

        size_t actual_size = inputs_list[i].size();
        for (auto& dim : input_shape)
            if (dim == -1) dim = static_cast<int64_t>(actual_size);

        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            allocator.GetInfo(),
            const_cast<float*>(inputs_list[i].data()),
            actual_size,
            input_shape.data(),
            input_shape.size()
        );

        input_tensors.push_back(std::move(tensor));
    }

    try {
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size()
        );

        return output_tensors;
    }
    catch (const Ort::Exception& e) {
        std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;
        throw; // rethrow or handle gracefully
    }

    return input_tensors;
}

std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>>
extract_bounding_boxes(
    const std::vector<std::vector<float>>& predictions,
    bool has_objectness,
    float confidence_threshold) {
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

    return {boxes, confidences, class_ids};
}

std::tuple<std::string, std::vector<Detection>>
postprocess_model_output(
    Ort::Session& onnx_session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const std::vector<const char*>& input_names,
    const std::vector<const char*>& output_names,
    const std::vector<std::vector<float>>& inputs_list,
    float confidence_threshold,
    float nms_threshold,
    rclcpp::Logger logger) {
    // Run ONNX postprocess
    auto onnx_results = execute_onnx_postprocess(
        onnx_session,
        allocator,
        input_names,
        output_names,
        inputs_list
    );

    std::vector<Detection> final_detections;
    std::string box_type;

    for (const auto& result : onnx_results) {
        auto shape_info = result.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = shape_info.GetShape();
        const float* data = result.GetTensorData<float>();

        int64_t N = 0, stride = 0;
        bool has_objectness = false;

        if (shape.size() == 3 && shape[2] == 85) {
            // YOLOv5
            N = shape[1];
            stride = 85;
            has_objectness = true;
            box_type = "xyxy";
        }
        else if (shape.size() == 3 && shape[1] == 84) {
            // YOLOv8
            N = shape[2];
            stride = 84;
            has_objectness = false;
            box_type = "xyxy";
        } else {
            throw std::runtime_error("Unexpected result shape");
        }

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        boxes.reserve(N);
        confidences.reserve(N);
        class_ids.reserve(N);

        // --- Parallelized parsing ---
        cv::parallel_for_(cv::Range(0, static_cast<int>(N)), [&](const cv::Range& range) {
            std::vector<cv::Rect> local_boxes;
            std::vector<float> local_confs;
            std::vector<int> local_classes;

            local_boxes.reserve(range.size());
            local_confs.reserve(range.size());
            local_classes.reserve(range.size());

            for (int i = range.start; i < range.end; ++i) {
                float x, y, w, h;
                const float* class_scores;

                if (has_objectness) {
                    // YOLOv5: [1, N, 85] format
                    const float* p = data + i * stride;
                    x = p[0];
                    y = p[1];
                    w = p[2];
                    h = p[3];
                    float obj = p[4];
                    class_scores = p + 5;

                    int best_class = 0;
                    float best_score = 0.0f;

                    for (int c = 0; c < 80; ++c) {  // 80 classes for COCO
                        float conf = obj * class_scores[c];
                        if (conf > best_score) {
                            best_score = conf;
                            best_class = c;
                        }
                    }

                    if (best_score >= confidence_threshold) {
                        int left = static_cast<int>(x - w / 2);
                        int top = static_cast<int>(y - h / 2);
                        int right = static_cast<int>(x + w / 2);
                        int bottom = static_cast<int>(y + h / 2);

                        local_boxes.emplace_back(left, top, right - left, bottom - top);
                        local_confs.push_back(best_score);
                        local_classes.push_back(best_class);
                    }
                } else {
                    // YOLOv8: [1, 84, N] format - channels are transposed
                    x = data[0 * N + i];  // Channel 0 for x coordinates
                    y = data[1 * N + i];  // Channel 1 for y coordinates
                    w = data[2 * N + i];  // Channel 2 for width
                    h = data[3 * N + i];  // Channel 3 for height

                    int best_class = 0;
                    float best_score = 0.0f;

                    // Class scores start from channel 4
                    for (int c = 0; c < 80; ++c) {  // 80 classes for COCO
                        float class_conf = data[(4 + c) * N + i];
                        if (class_conf > best_score) {
                            best_score = class_conf;
                            best_class = c;
                        }
                    }

                    if (best_score >= confidence_threshold) {
                        int left = static_cast<int>(x - w / 2);
                        int top = static_cast<int>(y - h / 2);
                        int right = static_cast<int>(x + w / 2);
                        int bottom = static_cast<int>(y + h / 2);

                        local_boxes.emplace_back(left, top, right - left, bottom - top);
                        local_confs.push_back(best_score);
                        local_classes.push_back(best_class);
                    }
                }
            }

            // Append local results (thread-safe with mutex or per-thread collection)
            static std::mutex mtx;
            std::lock_guard<std::mutex> lock(mtx);
            boxes.insert(boxes.end(), local_boxes.begin(), local_boxes.end());
            confidences.insert(confidences.end(), local_confs.begin(), local_confs.end());
            class_ids.insert(class_ids.end(), local_classes.begin(), local_classes.end());
        });

        // --- Apply Non-Maximum Suppression ---
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, indices);

        final_detections.reserve(final_detections.size() + indices.size());
        for (int idx : indices) {
            Detection det;
            det.class_id = class_ids[idx];
            det.confidence = confidences[idx];
            det.box = boxes[idx];
            final_detections.push_back(det);
        }
    }

    return {box_type, final_detections};
}

std::tuple<std::vector<std::int8_t>&, float, int, int> preprocess_frame(
    const cv::Mat& frame,
    const axrTensorInfo& info,
    const std::array<float, 3>& mean,
    const std::array<float, 3>& stddev,
    cv::Mat& padded_buffer,
    std::vector<std::int8_t>& quantized_buffer) {
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

    // Reuse padded buffer - clear and copy
    padded_buffer.setTo(cv::Scalar(0, 0, 0));
    resized.copyTo(padded_buffer(cv::Rect(x_offset, y_offset, resized_width, resized_height)));

    // Resize quantized buffer if needed
    const size_t total_size = height * width * channels;
    if (quantized_buffer.size() != total_size) {
        quantized_buffer.resize(total_size, static_cast<std::int8_t>(std::clamp(info.zero_point, -128, 127)));
    }

    // Precompute constants for maximum performance
    const std::array<float, 3> mul = {{1.0f / stddev[0], 1.0f / stddev[1], 1.0f / stddev[2]}};
    const std::array<float, 3> add = {{-mean[0] / stddev[0], -mean[1] / stddev[1], -mean[2] / stddev[2]}};
    const float inv_scale = 1.0f / info.scale;
    const float inv_255 = 1.0f / 255.0f;

    // Combined constants for each channel to minimize operations
    const float combined_mul[3] = {
        mul[0] * inv_scale * inv_255,
        mul[1] * inv_scale * inv_255,
        mul[2] * inv_scale * inv_255
    };
    const float combined_add[3] = {
        add[0] * inv_scale + info.zero_point,
        add[1] * inv_scale + info.zero_point,
        add[2] * inv_scale + info.zero_point
    };

    // Raw pointer access for maximum speed
    const uint8_t* src_ptr = padded_buffer.ptr<uint8_t>();
    int8_t* dst_ptr = quantized_buffer.data();

    // Optimized nested loop with combined operations
    for (int y = 0; y < static_cast<int>(height); ++y) {
        for (int x = 0; x < static_cast<int>(width); ++x) {
            const size_t pixel_idx = (y * width + x) * 3;

            // Process all 3 channels with combined operations (same math as simple version but faster)
            for (int c = 0; c < 3; ++c) {
                // Combined operation: pixel[2-c] * combined_mul[c] + combined_add[c]
                float qf = src_ptr[pixel_idx + (2 - c)] * combined_mul[c] + combined_add[c];
                float qf_clamped = std::clamp(qf, -128.0f, 127.0f);
                int8_t q = static_cast<int8_t>(std::round(qf_clamped));

                size_t idx = (y * width + x) * channels + c;
                dst_ptr[idx] = q;
            }
        }
    }

    return {quantized_buffer, scale, x_offset, y_offset};
}

cv::Mat plot_detections(
    const cv::Mat& frame,
    const std::vector<Detection>& detections,
    const std::string& box_type,
    const std::vector<std::string>& labels,
    const std::string& model_name,
    float scale,
    int x_offset,
    int y_offset) {
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
    const std::vector<std::string>& labels) {
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

auto read_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream file(path);
    for (std::string line; std::getline(file, line);) {
        labels.push_back(line);
    }
    return labels;
}

void logger(void* arg, axrLogLevel level, const char* msg) {
    (void)arg;
    (void)level;
}

// Define a ROS 2 Node class
class AxeleraYoloInference : public rclcpp::Node {
public:
    AxeleraYoloInference() : Node("axelera_yolo_inference") {
        RCLCPP_INFO(this->get_logger(), "Starting Axelera YOLO Inference Node...");

        // Declare parameters with default values
        this->declare_parameter("model_name", "");
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

        // Precompute mean and stddev for optimized preprocessing
        auto mean_param = this->get_parameter("mean").as_double_array();
        auto stddev_param = this->get_parameter("stddev").as_double_array();

        // Convert to arrays for faster access
        for (size_t i = 0; i < 3 && i < mean_param.size(); ++i) {
            mean_[i] = static_cast<float>(mean_param[i]);
        }
        for (size_t i = 0; i < 3 && i < stddev_param.size(); ++i) {
            stddev_[i] = static_cast<float>(stddev_param[i]);
        }

        // Load labels using default path
        const auto root = std::getenv("AXELERA_FRAMEWORK");
        this->labels_ = read_labels(root && root[0] != '\0' ? root + "/"s + DEFAULT_LABELS : DEFAULT_LABELS);
        onnx_model_path_ = "../build/" + model_name_ + "/" + model_name_ + "/1/postprocess_graph.onnx";

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

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        onnx_session_ = std::make_unique<Ort::Session>(env_, onnx_model_path_.c_str(), session_options);
        allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();

        // Query input/output count from the ONNX model
        size_t onnx_input_count = onnx_session_->GetInputCount();
        size_t onnx_output_count = onnx_session_->GetOutputCount();

        for (size_t i = 0; i < onnx_input_count; ++i) {
            Ort::AllocatedStringPtr name = onnx_session_->GetInputNameAllocated(i, *allocator_);
            input_names_.push_back(name.get());
            owned_input_names_.push_back(std::move(name));
        }

        for (size_t i = 0; i < onnx_output_count; ++i) {
            Ort::AllocatedStringPtr name = onnx_session_->GetOutputNameAllocated(i, *allocator_);
            output_names_.push_back(name.get());
            owned_output_names_.push_back(std::move(name));
        }

        // Precompute preprocessing parameters from input tensor info
        const auto& input_info = input_infos_[0];
        input_height_ = input_info.dims[1];
        input_width_ = input_info.dims[2];
        input_channels_ = input_info.dims[3];
        y_pad_left_ = input_info.padding[1][0];
        y_pad_right_ = input_info.padding[1][1];
        x_pad_left_ = input_info.padding[2][0];
        x_pad_right_ = input_info.padding[2][1];
        unpadded_height_ = input_height_ - y_pad_left_ - y_pad_right_;
        unpadded_width_ = input_width_ - x_pad_left_ - x_pad_right_;

        // Precompute normalization arrays
        normalization_mul_[0] = 1.0f / stddev_[0];
        normalization_mul_[1] = 1.0f / stddev_[1];
        normalization_mul_[2] = 1.0f / stddev_[2];
        normalization_add_[0] = -mean_[0] / stddev_[0];
        normalization_add_[1] = -mean_[1] / stddev_[1];
        normalization_add_[2] = -mean_[2] / stddev_[2];

        // Store quantization parameters
        input_scale_ = input_info.scale;
        input_zero_point_ = input_info.zero_point;
        quantized_buffer_size_ = input_height_ * input_width_ * input_channels_;

        // Initialize preallocated padded image buffer
        padded_buffer_ = cv::Mat(input_height_, input_width_, CV_8UC3);

        // Preallocated buffer for quantized data to avoid repeated allocations
        quantized_buffer_.resize(quantized_buffer_size_);

        // Precompute output processing parameters
        const auto& output_info = output_infos_[0];
        output_unpadded_shape_ = compute_unpadded_shape(output_info.dims, output_info.padding, output_info.ndims);
        output_N_ = output_unpadded_shape_[0];
        output_H_ = output_unpadded_shape_[1];
        output_W_ = output_unpadded_shape_[2];
        output_C_ = output_unpadded_shape_[3];
        output_padded_H_ = output_info.dims[1];
        output_padded_W_ = output_info.dims[2];
        output_padded_C_ = output_info.dims[3];
        output_input_stride_n_ = output_padded_H_ * output_padded_W_ * output_padded_C_;
        output_input_stride_h_ = output_padded_W_ * output_padded_C_;
        output_input_stride_w_ = output_padded_C_;
        output_output_stride_n_ = output_C_ * output_H_ * output_W_;
        output_output_stride_c_ = output_H_ * output_W_;
        output_output_stride_h_ = output_W_;
        output_pad_n_ = output_info.padding[0][0];
        output_pad_h_ = output_info.padding[1][0];
        output_pad_w_ = output_info.padding[2][0];
        output_pad_c_ = output_info.padding[3][0];
        output_scale_ = output_info.scale;
        output_zero_point_f_ = static_cast<float>(output_info.zero_point);
        output_result_size_ = output_N_ * output_C_ * output_H_ * output_W_;

        // Precomputed parameters for all outputs
        for (const auto& output_info : output_infos_) {
            OutputParams params;
            params.unpadded_shape = compute_unpadded_shape(output_info.dims, output_info.padding, output_info.ndims);
            params.N = params.unpadded_shape[0];
            params.H = params.unpadded_shape[1];
            params.W = params.unpadded_shape[2];
            params.C = params.unpadded_shape[3];
            params.padded_H = output_info.dims[1];
            params.padded_W = output_info.dims[2];
            params.padded_C = output_info.dims[3];
            params.input_stride_n = params.padded_H * params.padded_W * params.padded_C;
            params.input_stride_h = params.padded_W * params.padded_C;
            params.input_stride_w = params.padded_C;
            params.output_stride_n = params.C * params.H * params.W;
            params.output_stride_c = params.H * params.W;
            params.output_stride_h = params.W;
            params.pad_n = output_info.padding[0][0];
            params.pad_h = output_info.padding[1][0];
            params.pad_w = output_info.padding[2][0];
            params.pad_c = output_info.padding[3][0];
            params.scale = output_info.scale;
            params.zero_point_f = static_cast<float>(output_info.zero_point);
            params.result_size = params.N * params.C * params.H * params.W;
            all_output_params_.push_back(params);
        }

        RCLCPP_INFO(this->get_logger(), "Axelera YOLO Inference Node initialized successfully.");
    }

    ~AxeleraYoloInference() {
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
    std::string onnx_model_path_;

    std::unique_ptr<Ort::Session> onnx_session_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
    Ort::Env env_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<Ort::AllocatedStringPtr> owned_input_names_;
    std::vector<Ort::AllocatedStringPtr> owned_output_names_;

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

    // Precomputed preprocessing parameters (moved from preprocess_frame)
    size_t input_height_;
    size_t input_width_;
    size_t input_channels_;
    size_t y_pad_left_, y_pad_right_;
    size_t x_pad_left_, x_pad_right_;
    size_t unpadded_height_, unpadded_width_;
    std::array<float, 3> normalization_mul_;
    std::array<float, 3> normalization_add_;
    float input_scale_;
    float input_zero_point_;
    size_t quantized_buffer_size_;

    // Preallocated buffer for padded image
    cv::Mat padded_buffer_;

    // Preallocated buffer for quantized data to avoid repeated allocations
    std::vector<std::int8_t> quantized_buffer_;

    // Precomputed output processing parameters as simple variables (assuming single output for simplicity)
    std::vector<size_t> output_unpadded_shape_;
    size_t output_N_, output_H_, output_W_, output_C_;
    size_t output_padded_H_, output_padded_W_, output_padded_C_;
    size_t output_input_stride_n_, output_input_stride_h_, output_input_stride_w_;
    size_t output_output_stride_n_, output_output_stride_c_, output_output_stride_h_;
    size_t output_pad_n_, output_pad_h_, output_pad_w_, output_pad_c_;
    float output_scale_, output_zero_point_f_;
    size_t output_result_size_;

    // Precomputed parameters for all outputs
    struct OutputParams {
        std::vector<size_t> unpadded_shape;
        size_t N, H, W, C;
        size_t padded_H, padded_W, padded_C;
        size_t input_stride_n, input_stride_h, input_stride_w;
        size_t output_stride_n, output_stride_c, output_stride_h;
        size_t pad_n, pad_h, pad_w, pad_c;
        float scale, zero_point_f;
        size_t result_size;
    };
    std::vector<OutputParams> all_output_params_;

    std::vector<std::vector<float>> process_outputs(
        const std::vector<std::unique_ptr<std::int8_t[]>>& outputs) {
        std::vector<std::vector<float>> outs(outputs.size());

        // Parallelize over outputs using OpenCV's parallel_for_
        cv::parallel_for_(cv::Range(0, static_cast<int>(outputs.size())),
                          [&](const cv::Range& range) {
                              for (int i = range.start; i < range.end; ++i) {
                                  const int8_t* output = outputs[i].get();
                                  const auto& params = all_output_params_[i];

                                  std::vector<float> result(params.result_size);
                                  float* result_ptr = result.data();

                                  for (size_t n = 0; n < params.N; ++n) {
                                      const size_t n_in_base = (n + params.pad_n) * params.input_stride_n;
                                      const size_t n_out_base = n * params.output_stride_n;

                                      for (size_t h = 0; h < params.H; ++h) {
                                          const size_t h_in_base = n_in_base + (h + params.pad_h) * params.input_stride_h;
                                          const size_t h_out_base = n_out_base + h * params.output_stride_h;

                                          for (size_t w = 0; w < params.W; ++w) {
                                              const size_t w_in_base = h_in_base + (w + params.pad_w) * params.input_stride_w + params.pad_c;
                                              const size_t w_out_base = h_out_base + w;

                                              const int8_t* input_row = output + w_in_base;

                                              // Vectorized processing of channels (4 at a time)
                                              size_t c = 0;
                                              for (; c + 3 < params.C; c += 4) {
                                                  const int8_t v0 = input_row[c];
                                                  const int8_t v1 = input_row[c + 1];
                                                  const int8_t v2 = input_row[c + 2];
                                                  const int8_t v3 = input_row[c + 3];

                                                  result_ptr[w_out_base + c * params.output_stride_c] =
                                                      (v0 - params.zero_point_f) * params.scale;
                                                  result_ptr[w_out_base + (c + 1) * params.output_stride_c] =
                                                      (v1 - params.zero_point_f) * params.scale;
                                                  result_ptr[w_out_base + (c + 2) * params.output_stride_c] =
                                                      (v2 - params.zero_point_f) * params.scale;
                                                  result_ptr[w_out_base + (c + 3) * params.output_stride_c] =
                                                      (v3 - params.zero_point_f) * params.scale;
                                              }

                                              // Handle remaining channels
                                              for (; c < params.C; ++c) {
                                                  const int8_t v = input_row[c];
                                                  result_ptr[w_out_base + c * params.output_stride_c] =
                                                      (v - params.zero_point_f) * params.scale;
                                              }
                                          }
                                      }
                                  }

                                  // Store the result safely
                                  outs[i] = std::move(result);
                              }
                          });

        return outs;
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
            process_image(frame);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void process_image(const cv::Mat& frame) {
        auto [processed_input, scale, x_offset, y_offset] = preprocess_frame(
            frame, input_infos_[0], mean_, stddev_, padded_buffer_, quantized_buffer_);
        std::memcpy(input_args_[0].ptr, processed_input.data(), processed_input.size());

        // Model inference
        if (axr_run_model_instance(instance_, input_args_.data(), input_args_.size(),
                                   output_args_.data(), output_args_.size()) != AXR_SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Failed to run model instance");
            return;
        }

        auto dequantized_outputs = process_outputs(output_data_);

        auto [box_type, detections] = postprocess_model_output(
            *onnx_session_, *allocator_, input_names_, output_names_,
            dequantized_outputs, confidence_threshold_, nms_threshold_,
            this->get_logger());

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

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AxeleraYoloInference>());
    rclcpp::shutdown();
    return 0;
}
