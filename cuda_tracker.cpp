#include "siamese_tracker.h"
#include "utils.h"

extern MTimer timer;

__global__ void cuda_translate(float* dest_ptr, unsigned char* source_ptr,
    const int step, int cols, int rows, int channels){

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int _h = index / cols;
    int _w = index % cols;

    unsigned char *data_ptr = source_ptr + step * _h;
    data_ptr += channels * _w;

    dest_ptr[cols * rows * 0 + index] = data_ptr[0];
    dest_ptr[cols * rows * 1 + index] = data_ptr[1];
    dest_ptr[cols * rows * 2 + index] = data_ptr[2];
}

void SiameseTracker::cuda_fill_exemplar(float *exemplar_data, GpuMat exemplar){

    cuda_translate<<<exemplar.rows, exemplar.cols>>>(
        exemplar_data, exemplar.data, exemplar.step,
        exemplar.cols, exemplar.rows, exemplar.channels());
}

void SiameseTracker::cuda_prepare(cv::gpu::GpuMat &frame, Target* target, cv::Scalar channel_average, int target_index){

    boost::shared_ptr<Blob<float> > search_blob = caffe_net->blobs()[get_blob_index(string("crystal"))];

    int window_size = (int)std::round(target->search_range);

    cuda_crop_subwindow(frame, window_size, target, channel_average, search_input);

    float *data_ptr = VOLUME_SEARCH * target_index + search_blob->mutable_gpu_data();

    cuda_translate<<<search_input.rows, search_input.cols>>>(
            data_ptr, search_input.data, search_input.step,
            search_input.cols, search_input.rows, search_input.channels());
}

void SiameseTracker::cuda_crop_subwindow(GpuMat &key_frame, int window_size, Target *target, Scalar channel_average, GpuMat &input){

    input = channel_average;

    int new_size = input.cols;

    int pad_left = (int)std::round(window_size / 2.0 - target->box_position.x);
    int pad_right = (int)std::round(window_size / 2.0 + target->box_position.x - key_frame.cols);
    int pad_top = (int)std::round(window_size / 2.0 - target->box_position.y);
    int pad_bottom = (int)std::round(window_size / 2.0 + target->box_position.y - key_frame.rows);

    pad_left = MAX(pad_left, 0);
    pad_right = MAX(pad_right, 0);
    pad_top = MAX(pad_top, 0);
    pad_bottom = MAX(pad_bottom, 0);

    int crop_left = std::round(target->box_position.x - window_size / 2.0) + pad_left;
    int crop_top = std::round(target->box_position.y - window_size / 2.0) + pad_top;
    int crop_width = window_size - pad_left - pad_right;
    int crop_height = window_size - pad_top - pad_bottom;

    GpuMat crop = key_frame(Rect(crop_left, crop_top, crop_width, crop_height));

    float scale = (float)new_size / window_size;

    pad_left = (int)std::round(pad_left * scale);
    pad_top = (int)std::round(pad_top * scale);
    crop_width = (int)std::round(crop_width * scale);
    crop_height = (int)std::round(crop_height * scale);

    pad_left = MIN(new_size - crop_width, pad_left);
    pad_top = MIN(new_size - crop_height, pad_top);

    GpuMat dest = input(Rect(pad_left, pad_top, crop_width, crop_height));

    gpu::resize(crop, dest, dest.size());
}