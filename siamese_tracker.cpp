#include "siamese_tracker.h"
#include "utils.h"

extern MTimer timer;

extern string DEFINITION_FILE;
extern string MODEL_FILE;

float *SiameseTracker::hanning_vector(int length){
    float *vec = new float[length];
    for (int i=0; i<length; i++)
        vec[i] = HANN_ATTITUDE * (1 - cos(2 * M_PI * i / (length - 1)));
    return vec;
}

double SiameseTracker::max_value(Mat &image){

    double max_val;
    minMaxLoc(image, NULL, &max_val, NULL, NULL);

    return max_val;
}

double SiameseTracker::min_value(Mat &image){

    double min_val;
    minMaxLoc(image, &min_val, NULL, NULL, NULL);

    return min_val;
}

Point2f SiameseTracker::locate_peak(Mat &image){

    Point *max_loc = new Point();
    minMaxLoc(image, NULL, NULL, NULL, max_loc);

    Point2f peak_location(max_loc->x, max_loc->y);
    delete max_loc;

    return peak_location;
}

int SiameseTracker::get_blob_index(string blob_name){
    vector<string> blob_names = caffe_net->blob_names();
    int blob_index = -1;

    for(unsigned int i = 0; i < blob_names.size(); i++){
        if( blob_name == blob_names[i] ){
            blob_index = i;
            break;
        }
    }
    return blob_index;
}

void SiameseTracker::compute_stablizer(){
    stablizer_size = SIZE_SCORE * STRIDE;

    upsample = Mat(stablizer_size, stablizer_size, CV_32F);

    float *hanning_vec = hanning_vector(stablizer_size);

    Mat hanning_mat = Mat(stablizer_size, 1, CV_32F, hanning_vec);

    stablizer = hanning_mat * hanning_mat.t();
    stablizer /= sum(stablizer)[0];

    delete hanning_vec;
}

void SiameseTracker::compute_range(Target *target){
    float context_width = target->box_width + target->box_width / 2 + target->box_height / 2;
    float context_height = target->box_height + target->box_width / 2 + target->box_height / 2;

    target->exemplar_range = sqrt(context_width * context_height);
    target->search_range = target->exemplar_range * SIZE_SEARCH / SIZE_EXEMPLAR;

    target->lower_range_limit = target->search_range / SCALING_LIMIT;
    target->higher_range_limit = target->search_range * SCALING_LIMIT;
}

void SiameseTracker::delete_targets(){

    for(int i=0; i<targets.size(); i++){
        
        for(int j=0; j<targets[i].size(); j++)
            delete targets[i][j];
        
        targets[i].clear();
    }
    targets.clear();
}

SiameseTracker::SiameseTracker(){
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);

    string prototxt = string(WORK_DIR) + string(MODEL_DIR) + string(DEFINITION_FILE);
    string model = string(WORK_DIR) + string(MODEL_DIR) + string(MODEL_FILE);

    caffe_net.reset(new Net<float>(prototxt, TEST));
    caffe_net->CopyTrainedLayersFrom(model);

    compute_stablizer();

    search_input = GpuMat(SIZE_SEARCH, SIZE_SEARCH, CV_8UC3);

    exemplar_input = GpuMat(SIZE_EXEMPLAR, SIZE_EXEMPLAR, CV_8UC3);
}

SiameseTracker::~SiameseTracker(){
    delete_targets();
}
/*
bool SiameseTracker::init(const std::string &modelPath){

    utils::FileResourceLoader *pLoader = new utils::FileResourceLoader(modelPath.c_str());
    if(!pLoader)
        return false;
    utils::TarResourceLoader *pTarLoader = new utils::TarResourceLoader(pLoader);
    pTarLoader->BuildList();

    std::string prototxt = "model/siamese.prototxt";
    std::string caffemodel = "model/siamese.caffemodel";

    utils::TarFile *confFile = pTarLoader->FindFileByName("parameters.txt");
    if(!confFile)
        return false;

    int nSize = confFile->GetSize();
    char *buf = new char[nSize + 1];
    memset(buf, 0, nSize * sizeof(char));

    confFile->Read(buf, nSize, nSize);
    istringstream text(buf);

    if(!read_custom_config(text)){
        delete[] buf;
        return false;
    }
    delete[] buf;

    utils::TarFile * fileProto = pTarLoader->FindFileByName(prototxt.c_str());
    utils::TarFile * fileModel = pTarLoader->FindFileByName(caffemodel.c_str());

    caffe::NetParameter testNetParam;
    ReadProtoFromTextResource(*fileProto, &testNetParam);
    caffe::NetParameter trainedNetParam;
    ReadProtoFromBinResource(*fileModel, &trainedNetParam);

    caffe_net.reset(new caffe::Net<float>(testNetParam));
    caffe_net->CopyTrainedLayersFrom(trainedNetParam);

    delete pLoader;
    delete pTarLoader;
    return true;
}

bool SiameseTracker::read_custom_config(istringstream &fin){

    string strKey;
    int keys_found = 0;
    
    while(fin >> strKey){
        if (strKey.compare("SIZE_EXEMPLAR") == 0){
            fin >> SIZE_EXEMPLAR;
            keys_found++;
        }
        else if(strKey.compare("SIZE_SEARCH") == 0){
            fin >> SIZE_SEARCH;
            keys_found++;
        }
        else if(strKey.compare("SIZE_SCORE") == 0){
            fin >> SIZE_SCORE;
            keys_found++;
        }
        else if(strKey.compare("STRIDE") == 0){
            fin >> STRIDE;
            keys_found++;
        }
        else if(strKey.compare("VOLUME_EXEMPLAR") == 0){
            fin >> VOLUME_EXEMPLAR;
            keys_found++;
        }
        else if(strKey.compare("VOLUME_SEARCH") == 0){
            fin >> VOLUME_SEARCH;
            keys_found++;
        }
        else if(strKey.compare("VOLUME_SCORE") == 0){
            fin >> VOLUME_SCORE;
            keys_found++;
        }
        else if(strKey.compare("NUM_SCALES") == 0){
            fin >> NUM_SCALES;
            keys_found++;
        }
        else if(strKey.compare("SCALING_STEP") == 0){
            fin >> SCALING_STEP;
            keys_found++;
        }
        else if(strKey.compare("SCALING_PENALTY") == 0){
            fin >> SCALING_PENALTY;
            keys_found++;
        }
        else if(strKey.compare("SCALING_LIMIT") == 0){
            fin >> SCALING_LIMIT;
            keys_found++;
        }
        else if(strKey.compare("STABLIZER_WEIGHT") == 0){
            fin >> STABLIZER_WEIGHT;
            keys_found++;
        }
        else if(strKey.compare("ADAPTION_RATE") == 0){
            fin >> ADAPTION_RATE;
            keys_found++;
        }
        else if(strKey.compare("HANN_ATTITUDE") == 0){
            fin >> HANN_ATTITUDE;
            keys_found++;
        }
    }
    return keys_found == 14;
}
*/

void SiameseTracker::reset(vector<GpuMat> &key_frame, vector<vector<Rect*> > &key_frame_boxes){
    
    delete_targets();

    camera_channel_averages.clear();

    total_targets = 0;

    for(int i=0; i<key_frame_boxes.size(); i++)
        total_targets += key_frame_boxes[i].size();

    boost::shared_ptr<Blob<float> > exemplar_blob = caffe_net->blobs()[get_blob_index(string("kernel"))];
    boost::shared_ptr<Blob<float> > search_blob = caffe_net->blobs()[get_blob_index(string("crystal"))];

    exemplar_blob->Reshape(total_targets, 3, SIZE_EXEMPLAR, SIZE_EXEMPLAR);
    search_blob->Reshape(total_targets, 3, SIZE_SEARCH, SIZE_SEARCH);

    caffe_net->Reshape();

    assert(exemplar_blob->count() == total_targets * VOLUME_EXEMPLAR);
    assert(search_blob->count() == total_targets * VOLUME_SEARCH);

    int data_ptr = 0;

    for(int i=0; i<key_frame_boxes.size(); i++){

        targets.push_back(vector<Target*>());

        GpuMat cuda_frame = key_frame[i];

        // key_frame[i].convertTo(cuda_frame, CV_32FC3);

        Scalar sum = gpu::sum(cuda_frame);
        float avg0 = sum[0] / cuda_frame.cols / cuda_frame.rows;
        float avg1 = sum[1] / cuda_frame.cols / cuda_frame.rows;
        float avg2 = sum[2] / cuda_frame.cols / cuda_frame.rows;
        camera_channel_averages.push_back(Scalar(avg0, avg1, avg2));

        for(int j=0; j<key_frame_boxes[i].size(); j++){

            targets[i].push_back(new Target());

            Target *target = targets[i][j];
            Rect* box = key_frame_boxes[i][j];

            target->box_width = box->width;
            target->box_height = box->height;

            float center_x = box->x + target->box_width / 2;
            float center_y = box->y + target->box_height / 2;

            target->box_position = Point2f(center_x, center_y);

            compute_range(target);

            int window_size = (int)std::round(target->exemplar_range);

            cuda_crop_subwindow(cuda_frame, window_size, target, camera_channel_averages[i], exemplar_input);

            cuda_fill_exemplar(exemplar_blob->mutable_gpu_data() + data_ptr, exemplar_input);

            data_ptr += VOLUME_EXEMPLAR;
        }
    }
}

Rect *SiameseTracker::regress(Mat &response, Target *target){

    resize(response, upsample, upsample.size());

    upsample -= min_value(upsample);

    upsample /= sum(upsample)[0];

    addWeighted(upsample, 1 - STABLIZER_WEIGHT, stablizer, STABLIZER_WEIGHT, 0, upsample);

    Point2f displacement = locate_peak(upsample);

    displacement.x -= stablizer_size / 2;
    displacement.y -= stablizer_size / 2;

    target->box_position.x += displacement.x * target->search_range / SIZE_SEARCH;
    target->box_position.y += displacement.y * target->search_range / SIZE_SEARCH;

    int anchor_left = (int)std::round(target->box_position.x - target->box_width / 2);
    int anchor_top = (int)std::round(target->box_position.y - target->box_height / 2);

    return new Rect(anchor_left, anchor_top, (int)std::round(target->box_width), (int)std::round(target->box_height));
}

vector<vector<Rect*> > SiameseTracker::track(vector<GpuMat> &cameras){

    int target_index = 0;

    for(int ci=0; ci<cameras.size(); ci++){

        GpuMat cuda_frame = cameras[ci];

        // cameras[ci].convertTo(cuda_frame, CV_32FC3);

        for(int ti=0; ti<targets[ci].size(); ti++){

            Target *this_target = targets[ci][ti];

            cuda_prepare(cuda_frame, this_target, camera_channel_averages[ci], target_index + ti);
        }
    }
    cudaDeviceSynchronize();

    // cin.get();
    timer.end("before forward: ");

    caffe_net->ForwardPrefilled();

    boost::shared_ptr<Blob<float> > response_blob = caffe_net->blobs()[get_blob_index(string("chance"))];

    float *response_data = const_cast<float*>(response_blob->cpu_data());

    vector<vector<Mat> > response_maps;

    int data_ptr = 0;

    // cin.get();
    timer.end("after forward: ");

    for(int i=0; i<targets.size(); i++){

        response_maps.push_back(vector<Mat>());

        for(int j=0; j<targets[i].size(); j++){

            response_maps[i].push_back(Mat(SIZE_SCORE, SIZE_SCORE, CV_32F, response_data + data_ptr));

            data_ptr += VOLUME_SCORE;
        }
    }
    vector<vector<Rect*> > regressed_boxes;

    for(int i=0; i<cameras.size(); i++){

        regressed_boxes.push_back(vector<Rect*>());

        for(int j=0; j<targets[i].size(); j++){

            Mat &this_map = response_maps[i][j];

            Target *this_target = targets[i][j];
            
            regressed_boxes[i].push_back(regress(this_map, this_target));
        }
    }
    return regressed_boxes;
}