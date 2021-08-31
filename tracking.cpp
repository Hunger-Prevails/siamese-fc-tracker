#include "siamese_tracker.h"
#include "utils.h"
#include <time.h>

#include <fstream>

Rect *to_rect(string line){
	
	float x, y, width, height;

	stringstream sin;
	sin << line;

	sin >> x;
	sin.seekg(1, ios::cur);
	sin >> y;
	sin.seekg(1, ios::cur);
	sin >> width;
	sin.seekg(1, ios::cur);
	sin >> height;

    return new Rect(x-1, y-1, width, height);
}

MTimer timer;

string VIDEO_FILE;
string VIDEO_DIR;
string DEFINITION_FILE;
string MODEL_FILE;

int main(){

	string input_catalogue = string(WORK_DIR) + string(INPUT_CATALOGUE);

	ifstream input(input_catalogue);

	input >> VIDEO_FILE;
	input >> VIDEO_DIR;
	input >> DEFINITION_FILE;
	input >> MODEL_FILE;

	input.close();

	SiameseTracker *tracker = new SiameseTracker();

	ImageRetriever *retriever = new ImageRetriever(100);

	string ground_truth = string(WORK_DIR) + string(VIDEO_DIR) + string(GROUND_TRUTH);

	ifstream fin(ground_truth);

	vector<vector<Rect*> > initial_boxes;

	vector<Rect*> this_camera;

	string line;

	while(fin >> line)
		this_camera.push_back(to_rect(line));

	initial_boxes.push_back(this_camera);

	string video_file = string(WORK_DIR) + string(VIDEO_DIR) + string(VIDEO_FILE);

	VideoCapture capture(video_file);
	assert(capture.isOpened());

	Mat frame;
	capture >> frame;

	vector<GpuMat> initial_frames;

	initial_frames.push_back(GpuMat(frame));

	tracker->reset(initial_frames, initial_boxes);

	int count = 0;

	while(true){
		capture >> frame;

		if (frame.empty()){

			cout << "Tracking Over" << endl;
			break;
		}
		vector<GpuMat> current_frame;

		current_frame.push_back(GpuMat(frame));
		
		timer.start();

		vector<vector<Rect*> > track_boxes = tracker->track(current_frame);

		// cin.get();
		timer.end("after track: ");

		this_camera = track_boxes[0];

		for(int i = 0; i < this_camera.size(); i++)
			rectangle(frame, *this_camera[i], GREEN, LINE_WIDTH);

		retriever->save_image(++count, frame);

		if(count == 50)
			break;
	}
	return 0;
}