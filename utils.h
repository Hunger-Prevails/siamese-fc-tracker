#include "definitions.h"

class ImageRetriever{

public:
    ImageRetriever(int count_frames);
    Mat next_frame();
    void save_image(int index, Mat frame);
    stringstream ss;

private:
    int count_frames;
    int frame_index;
};

class MTimer{

private:
    clock_t mark;
public:
    MTimer();
    void start();
    void end(string message);
};
