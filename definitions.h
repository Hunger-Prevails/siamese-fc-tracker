#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <memory>
#include <limits>

#include "opencv2/gpu/gpu.hpp"
#include "opencv2/opencv.hpp"
#include "caffe/caffe.hpp"

using namespace std;
using namespace cv;
using namespace gpu;
using namespace caffe;

#define SIZE_EXEMPLAR 127
#define SIZE_SEARCH 255
#define SIZE_SCORE 17
#define STRIDE 8
#define VOLUME_EXEMPLAR 3 * SIZE_EXEMPLAR * SIZE_EXEMPLAR
#define VOLUME_SEARCH 3 * SIZE_SEARCH * SIZE_SEARCH
#define VOLUME_SCORE 17 * 17

#define START_FRAME_INDEX 0

#define STABLIZER_WEIGHT 0.176
#define HANN_ATTITUDE 0.5
#define ADAPTION_RATE 0.59

#define WORK_DIR "/home/sensetime/Documents/Siamese-MOT/"
#define GROUND_TRUTH "groundtruth.txt"
#define MODEL_DIR "caffe_model/"
#define FILE_SUFFIX ".jpg"
#define INPUT_CATALOGUE "input.txt"

#define GREEN Scalar(0, 255, 0)
#define RED Scalar(0, 0, 255)
#define LINE_WIDTH 2

#define SCALING_STEP 1.0375
#define SCALING_PENALTY 0.9745
#define SCALING_LIMIT 5