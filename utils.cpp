#include "utils.h"

extern string VIDEO_DIR;

ImageRetriever::ImageRetriever(int frame_count){
    count_frames = frame_count;
    frame_index = START_FRAME_INDEX;
}

Mat ImageRetriever::next_frame(){
    if(frame_index == count_frames)
        return Mat();

    ss.clear();
    ss << setfill('0') << setw(8) << ++frame_index;
    string filename;
    ss >> filename;

    filename = string(WORK_DIR) + string(VIDEO_DIR) + filename + string(FILE_SUFFIX);
    Mat dest = imread(filename);
    dest.convertTo(dest, CV_32FC3);
    return dest;
}

void ImageRetriever::save_image(int index, Mat frame){
    ss.clear();
    ss << "outcome_" << index;
    string filename;
    ss >> filename;

    filename = string(WORK_DIR) + string(VIDEO_DIR) + filename + string(FILE_SUFFIX);
    imwrite(filename, frame);
}

MTimer::MTimer(){

}

void MTimer::start(){
    mark = clock();
}

void MTimer::end(string message){
    clock_t current = clock();
    cout << message << "(" << (current - mark) * 1000.0 / CLOCKS_PER_SEC << ")" << endl;
    mark = current;
}
