#include "crestereo.hpp"
#include <iostream>
#include <unistd.h>
#include <spdlog/spdlog.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "tensorrt_utils/buffers.h"
#include "tensorrt_utils/logger.h"
#include "tensorrt_utils/common.h"

namespace TensorRTCrestereo{
int32_t CrestereoTrt::init(const std::string& onnx_model_path, const std::string& trt_engine_path, int32_t stream_number){
  initLibNvInferPlugins(&tensorrt_log::gLogger, "");
  if(stream_number <= 0){
    stream_number = 1;
  }
  stream_number_ = stream_number;

  //Check TRT engine file can be loaded, if not create engine from onnx model and save to trt_engine_path
  if (access(trt_engine_path.c_str(), F_OK) == -1){
    spdlog::info("TRT engine file not found, create engine from onnx model");
    if (access(onnx_model_path.c_str(), F_OK) == -1){
      spdlog::error("onnx model file not found");
      return -2;
    }
    if (buildEngine(onnx_model_path, trt_engine_path) != 0){
      spdlog::error("buildEngine failed");
      return -3;
    }
  } else {
    spdlog::info("TRT engine file found, load engine from file");
    if (deserializeEngine(trt_engine_path) != 0){
      spdlog::error("deserializeEngine failed");
      return -4;
    }
  }

  //Create executors
  for(int32_t i = 0; i < stream_number_; i++){
    CrestereoExcutor excutor;
    int32_t ret = excutor.init(this->nv_engine_ptr_,"","");
    if(ret != 0){
      spdlog::error("Crestereo multistream Excutor init failed");
      return -5;
    }
    this->executors_.push_back(std::move(excutor));
  }
  printf("Success to init CrestereoTrt\n");
  return 0;
}

//TODO: inference stucked 
int32_t CrestereoTrt::doInference(const cv::Mat input[4]){  
  for(int32_t i = 0; i < this->stream_number_; i++){
    if(input[i].empty()){
      return 0;
    }
    int32_t ret = this->executors_[i].setInputImages(input[i]);
    if(ret != 0){
      std::cout << "setInputImages failed" << std::endl;
      return -2;
    }
  }

  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i].doInference();
    if(ret != 0){
      std::cout << "doInference failed" << std::endl;
      return -3;
    }
  }

  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i].copyBack();
    if(ret != 0){
      std::cout << "doInference failed" << std::endl;
      return -3;
    }
  }

  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i].synchronize();
    if(ret != 0){
      std::cout << "synchronize failed" << std::endl;
      return -4;
    }
  }

  #ifdef DEBUG
  printf ("inferenced\n");
  #endif

  return 0;
}

int32_t CrestereoTrt::doInference(const cv::cuda::GpuMat input[4][2]){  
  for(int32_t i = 0; i < this->stream_number_; i++){
    if(input[i][0].empty()){
      return 0;
    }
    int32_t ret = this->executors_[i].setInputImages(input[i][0],input[i][1]);
    if(ret != 0){
      std::cout << "setInputImages failed" << std::endl;
      return -2;
    }
  }

  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i].doInference();
    if(ret != 0){
      std::cout << "doInference failed" << std::endl;
      return -3;
    }
  }

  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i].copyBack();
    if(ret != 0){
      std::cout << "doInference failed" << std::endl;
      return -3;
    }
  }

  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i].synchronize();
    if(ret != 0){
      std::cout << "synchronize failed" << std::endl;
      return -4;
    }
  }

  #ifdef DEBUG
  printf ("inferenced\n");
  #endif

  return 0;
}

int32_t CrestereoTrt::getOutput(cv::Mat output[4]){
  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i].getOutput(output[i]);
    if(ret != 0){
      std::cout << "getOutput failed" << std::endl;
      return -5;
    }
  }
  return 0;
}

CrestereoTrt::~CrestereoTrt(){
  for(int32_t i = 0; i < this->stream_number_; i++){
    this->executors_[i].~CrestereoExcutor();
  }
  usleep(100);
  this->nv_engine_ptr_->destroy();  //TODO: deconstruct bug here
}


int32_t CrestereoTrt::deserializeEngine(const std::string& trt_engine_path){
  std::ifstream engine_file(trt_engine_path.c_str(), std::ios::binary);
  if (engine_file.is_open()){
    spdlog::info("load engine from file: {}", trt_engine_path);
    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    char * engine_data = new char[size];
    engine_file.read(engine_data, size);
    engine_file.close();
    IRuntime* nv_runtime = createInferRuntime(tensorrt_log::gLogger);
    this->nv_engine_ptr_ = std::shared_ptr<nvinfer1::ICudaEngine>(nv_runtime->deserializeCudaEngine(engine_data, size, nullptr));
    delete[] engine_data;
    if(this->nv_engine_ptr_ == nullptr){
      spdlog::error("deserializeCudaEngine failed");
      return -3;
    }
    spdlog::info("Success to load engine from file");
    return 0;
  } else {
    return -2;
  }
}

int32_t CrestereoTrt::buildEngine(const std::string& onnx_model_path, const std::string& trt_engine_path){
  auto builder = tensorrt_common::TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(tensorrt_log::gLogger.getTRTLogger()));
  if (builder == nullptr){
    spdlog::error("createInferBuilder failed");
    return -1;
  }
  auto network = tensorrt_common::TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
    1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  if (network == nullptr){
    spdlog::error("createNetworkV2 failed");
    return -2;
  }
  auto config = tensorrt_common::TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (config == nullptr){
    spdlog::error("createBuilderConfig failed");
    return -3;
  }
  auto parser = tensorrt_common::TensorRTUniquePtr<nvonnxparser::IParser>(
    nvonnxparser::createParser(*network, tensorrt_log::gLogger.getTRTLogger()));
  if (parser == nullptr){
    spdlog::error("createParser failed");
    return -4;
  }
  auto profile = builder->createOptimizationProfile();
  if (profile == nullptr){
    spdlog::error("createOptimizationProfile failed");
    return -5;
  }
  profile->setDimensions("left", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 240, 320});
  profile->setDimensions("right", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 240, 320});
  config->addOptimizationProfile(profile);

  auto parsed = parser->parseFromFile(onnx_model_path.c_str(), static_cast<int>(tensorrt_log::gLogger.getReportableSeverity()));
  if (!parsed){
    spdlog::error("parseFromFile failed");
    return -6;
  }
  config ->setMaxWorkspaceSize(4096_MiB);
  config ->setFlag(nvinfer1::BuilderFlag::kFP16);
  config ->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
  
  auto profile_stream = tensorrt_common::makeCudaStream();
  if (profile_stream == nullptr){
    spdlog::error("makeCudaStream failed");
    return -7;
  }
  config->setProfileStream(*profile_stream);
  tensorrt_common::TensorRTUniquePtr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (plan == nullptr){
    spdlog::error("buildSerializedNetwork failed");
    return -8;
  }
  tensorrt_common::TensorRTUniquePtr<IRuntime> runtime{createInferRuntime(tensorrt_log::gLogger.getTRTLogger())};
  if (runtime == nullptr){
    spdlog::error("createInferRuntime failed");
    return -9;
  }
  this->nv_engine_ptr_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size(), nullptr));
  if (this->nv_engine_ptr_ == nullptr){
    spdlog::error("deserializeCudaEngine failed");
    return -10;
  }

  //Save engine to file
  nvinfer1::IHostMemory* engine_data = this->nv_engine_ptr_->serialize();
  std::ofstream engine_file(trt_engine_path.c_str(), std::ios::binary);
  if (engine_file.is_open()){
    engine_file.write(static_cast<const char*>(engine_data->data()), engine_data->size());
    engine_file.close();
    spdlog::info("Success to save engine to file: {}", trt_engine_path);
  } else {
    spdlog::error("Failed to save engine to file: {}", trt_engine_path);
    return -11;
  }
  return 0;
}



int32_t CrestereoExcutor::init(std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr, 
  std::string input_tensor_name,
  std::string output_tensor_name){
  this->engine_ptr_ = engine_ptr;
  this->nv_context_ptr_ = this->engine_ptr_->createExecutionContext();
  if(this->nv_context_ptr_ == nullptr){
    std::cout << "createExecutionContext failed" << std::endl;
    return -1;
  }
  cudaStreamCreate(&this->stream_);
  this->cv_stream = cv::cuda::StreamAccessor::wrapStream(
          this->stream_
      );
  if(this->stream_ == nullptr){
    std::cout << "cudaStreamCreate failed" << std::endl;
    return -2;
  }
  this->buffer_manager_ptr_ = std::make_unique<tensorrt_buffer::BufferManager>(this->engine_ptr_, 
    0,this->nv_context_ptr_);
  if(this->buffer_manager_ptr_ == nullptr){
    std::cout << "make_unique BufferManager failed" << std::endl;
    return -3;
  }

  this->input_tensor_name1_ = "left";
  this->input_tensor_name2_ = "right";
  this->output_tensor_name_ = "output";
  this->input_index_ = this->engine_ptr_->getBindingIndex(this->input_tensor_name1_.c_str());
  if(this->input_index_ < 0){
    std::cout << "getBindingIndex failed" << std::endl;
    return -4;
  }
  auto input_dim = this->engine_ptr_->getBindingDimensions(this->input_index_);
  //Easy Coredump is donnot know your data type
  this->input_size_ = input_dim.d[0] * input_dim.d[1] * input_dim.d[2] * input_dim.d[3] * sizeof(float); 
  this->output_index_ = this->engine_ptr_->getBindingIndex(this->output_tensor_name_.c_str());
  if(this->output_index_  < 0){
    std::cout << "getBindingIndex failed" << std::endl;
    return -5;
  }
  auto output_dim = this->engine_ptr_->getBindingDimensions(this->output_index_ );
  this->output_size_ = output_dim.d[0] * output_dim.d[1] * output_dim.d[2] * output_dim.d[3] * sizeof(float);
  return 0;
}

int32_t CrestereoExcutor::setInputImages(const cv::Mat& input){
  cv::Mat left  = input(cv::Rect(0, 0, input.cols, input.rows /2 )).clone();
  cv::Mat right = input(cv::Rect(0, input.rows/2, input.cols, input.rows /2 )).clone();

//   memcpy(this->buffer_manager_ptr_->getHostBuffer(this->input_tensor_name1_), left.data, this->input_size_);
//   memcpy(this->buffer_manager_ptr_->getHostBuffer(this->input_tensor_name2_), right.data, this->input_size_);
    float* hostDataBuffer1 = static_cast<float*>(this->buffer_manager_ptr_->getHostBuffer(this->input_tensor_name1_));
    float* hostDataBuffer2 = static_cast<float*>(this->buffer_manager_ptr_->getHostBuffer(this->input_tensor_name2_));
    int channelSize = 240 * 320;

    std::vector<cv::Mat> leftChannels(3), rightChannels(3);
    cv::split(left, leftChannels);
    cv::split(right, rightChannels);

// 左图
for (int c = 0; c < 3; c++)
{
    memcpy(
        hostDataBuffer1 + c * channelSize,
        leftChannels[c].ptr<float>(0),
        channelSize * sizeof(float)
    );
}

// 右图
for (int c = 0; c < 3; c++)
{
    memcpy(
        hostDataBuffer2 + c * channelSize,
        rightChannels[c].ptr<float>(0),
        channelSize * sizeof(float)
    );
}
  buffer_manager_ptr_->copyInputToDeviceAsync(this->stream_);
  return 0;
}

int32_t CrestereoExcutor::setInputImages(const cv::cuda::GpuMat& left,const cv::cuda::GpuMat& right){
    float* deviceDataBuffer1 = static_cast<float*>(this->buffer_manager_ptr_->getDeviceBuffer(this->input_tensor_name1_));
    float* deviceDataBuffer2 = static_cast<float*>(this->buffer_manager_ptr_->getDeviceBuffer(this->input_tensor_name2_));
    int channelSize = 240 * 320;

    std::vector<cv::cuda::GpuMat> leftChannels(3), rightChannels(3);
    
    cv::cuda::split(left, leftChannels, this->cv_stream);
    cv::cuda::split(right, rightChannels,this->cv_stream);
    // 左图
    for (int c = 0; c < 3; c++)
    {
      size_t width_bytes = leftChannels[c].cols * leftChannels[c].elemSize();

      cudaMemcpy2DAsync(
          deviceDataBuffer1 + c * channelSize,
          leftChannels[c].cols * leftChannels[c].elemSize(), // dst pitch
          leftChannels[c].data,
          leftChannels[c].step,                   // src pitch
          width_bytes,
          leftChannels[c].rows,
          cudaMemcpyDeviceToDevice,
          this->stream_
      );
    }

    // // 右图
    for (int c = 0; c < 3; c++)
    {
      size_t width_bytes = rightChannels[c].cols * rightChannels[c].elemSize();

      cudaMemcpy2DAsync(
          deviceDataBuffer2 + c * channelSize,
          leftChannels[c].cols * rightChannels[c].elemSize(), // dst pitch
          rightChannels[c].data,
          rightChannels[c].step,                   // src pitch
          width_bytes,
          rightChannels[c].rows,
          cudaMemcpyDeviceToDevice,
          this->stream_
      );
    }
  return 0;
}

int32_t CrestereoExcutor::doInference(){
  bool status = this->nv_context_ptr_->enqueueV2(this->buffer_manager_ptr_->getDeviceBindings().data(), this->stream_, nullptr);
  if (!status){
    std::cout << "enqueueV2 failed" << std::endl;
    return -1;
  }
  return 0;
}

int32_t CrestereoExcutor::copyBack(){
  buffer_manager_ptr_->copyOutputToHostAsync(this->stream_);
  return 0;
}

int32_t CrestereoExcutor::synchronize(){
  return cudaStreamSynchronize(this->stream_);
}

int32_t CrestereoExcutor::getOutput(cv::Mat& output){
  memcpy(output.data, this->buffer_manager_ptr_->getHostBuffer(this->output_tensor_name_), this->output_size_ / 2);
  return 0;
}
}