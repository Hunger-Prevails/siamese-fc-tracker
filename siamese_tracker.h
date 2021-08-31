#include "definitions.h"

class Target{

public:
    cv::Point2f box_position;

    float box_width;
    float box_height;

    float exemplar_range;
    float search_range;
    float lower_range_limit;
    float higher_range_limit;
};

class SiameseTracker{

private:
    boost::shared_ptr<caffe::Net<float> > caffe_net;

    cv::gpu::GpuMat search_input;

    cv::gpu::GpuMat exemplar_input;

    cv::Mat stablizer;

    cv::Mat upsample;

    int stablizer_size;

    int total_targets;

    std::vector<cv::Scalar> camera_channel_averages;
    
    std::vector<std::vector<Target*> > targets;

private:
    float *hanning_vector(int length);

    double max_value(cv::Mat &image);

    double min_value(cv::Mat &image);

    cv::Point2f locate_peak(cv::Mat &image);

    void cuda_fill_exemplar(float *exemplar_data, cv::gpu::GpuMat exemplar);

    int get_blob_index(std::string blob_name);

    void delete_targets();

protected:
    void compute_range(Target *target);

    void compute_stablizer();

    void cuda_crop_subwindow(GpuMat &key_frame, int window_size, Target *target, Scalar channel_average, GpuMat &input);

    void cuda_prepare(cv::gpu::GpuMat &frame, Target* target, cv::Scalar channel_average, int target_index);

public:
    SiameseTracker();

    ~SiameseTracker();

    // bool init(const std::string &modelPath);

    // bool read_custom_config(std::istringstream &fin);

    void reset(std::vector<cv::gpu::GpuMat> &key_frame, std::vector<std::vector<cv::Rect*> > &key_frame_boxes);

    cv::Rect* regress(cv::Mat &response, Target *target);

    std::vector<std::vector<cv::Rect*> > track(std::vector<cv::gpu::GpuMat> &cameras);
};
