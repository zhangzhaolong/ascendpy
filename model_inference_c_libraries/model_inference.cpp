/**
 * ============================================================================
 *
 * Copyright (C) 2018, Hisilicon Technologies Co., Ltd. All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1 Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   2 Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   3 Neither the names of the copyright holders nor the names of the
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================
 */

#include "model_inference.h"

#include <memory>
#include <fstream>
#include <unistd.h>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <regex>
#include <securec.h>

using namespace std;

namespace ascend {
namespace modelinference {

namespace {
const int kHasNoAccessPermission = -1; // the file has no read permission
}

ModelInference::ModelInference() {
  model_path_ = "";
  image_path_ = "";
  image_width_ = 0;
  image_height_ = 0;
}

bool ModelInference::VerifyFilePath(const string &model_path) {
  if (model_path.empty()) {
    ASC_LOG_ERROR("the model path is empty!", model_path.c_str());
    return false;
  }

  if (access(model_path.c_str(), F_OK) == kHasNoAccessPermission) {
    ASC_LOG_ERROR("the model path:%s is not exist!", model_path.c_str());
    return false;
  }

  if (access(model_path.c_str(), R_OK) == kHasNoAccessPermission) {
    ASC_LOG_ERROR("the model path:%s has no read permission!",
                  model_path.c_str());
    return false;
  }

  return true;
}

void ModelInference::Init(const string &model_path) {
  ASC_LOG_INFO("start initialize ModelInference!");

  if (!VerifyFilePath(model_path)) {
    throw "fail to verify the input model path!";
  }

  if (ai_model_manager_ == nullptr) { // check ai model manager is nullptr
    ai_model_manager_ = make_shared<hiai::AIModelManager>();
  }

  vector<hiai::AIModelDescription> model_desc_vec;
  hiai::AIModelDescription model_description;

  model_path_ = model_path;
  model_description.set_path(model_path_);

  hiai::AIConfig config;
// check ai model manager initialize result
  if (ai_model_manager_->Init(config, model_desc_vec) != hiai::SUCCESS) {
    throw "fail to initialize the input model path!";
  }

  ASC_LOG_INFO("end initialize ModelInference!");
}

void ModelInference::Init(const string &model_path, int iamge_width,
                          int iamge_height) {
  if (!VerifyFilePath(model_path)) {
    throw "fail to verify the input model path!";
  }

  model_path_ = model_path;

  if (iamge_width < 1 || iamge_width > 4096 || iamge_height < 1
      || iamge_height > 4096) {
    ASC_LOG_ERROR(
        "the image width and height:[%d, %d] are invalid, value rang:1~4096",
        iamge_width, iamge_height);
    throw "the input image width and height are invalid!";
  }

  image_width_ = iamge_width;
  image_height_ = iamge_height;
}

void ModelInference::Inference(const char* input_buffer, int input_buffer_size,
                               float** output_buffer,
                               int** output_buffer_size) {
  vector<shared_ptr<hiai::IAITensor>> input_data_vec;
  vector<shared_ptr<hiai::IAITensor>> output_data_vec;

  ASC_LOG_INFO("start ai model inference, model path:%s!", model_path_.c_str());

  shared_ptr<hiai::AINeuralNetworkBuffer> neural_buffer = shared_ptr<
      hiai::AINeuralNetworkBuffer>(new (nothrow) hiai::AINeuralNetworkBuffer());
  if (neural_buffer.get() == nullptr) { // check new memory result
    ASC_LOG_ERROR("fail to new memory when initialize neural buffer!");
    throw "fail to new memory when initialize neural buffer!";
  }

  neural_buffer->SetBuffer((void*) (input_buffer), input_buffer_size);
  shared_ptr<hiai::IAITensor> input_data = static_pointer_cast<hiai::IAITensor>(
      neural_buffer);
  input_data_vec.push_back(input_data);

// Call Process, Predict
  if (ai_model_manager_->CreateOutputTensor(input_data_vec, output_data_vec)
      != hiai::SUCCESS) {
    ASC_LOG_ERROR("CreateOutputTensor failed");
    throw "fail to create output tensor!";
  }

  hiai::AIContext ai_context;
  ASC_LOG_INFO("start ai_model_manager_->Process!");
  hiai::AIStatus ret_process = ai_model_manager_->Process(ai_context,
                                                          input_data_vec,
                                                          output_data_vec, 0);
  if (ret_process != hiai::SUCCESS) {
    ASC_LOG_ERROR("ai_model_manager Process failed");
    throw "ai_model_manager Process failed!";
  }

  ASC_LOG_INFO("end ai_model_manager_->Process!!");

  **output_buffer_size = 0;
// loop for each data in output_data_vec
  for (int n = 0; n < output_data_vec.size(); ++n) {
    std::shared_ptr<hiai::AINeuralNetworkBuffer> result_tensor =
        std::static_pointer_cast<hiai::AINeuralNetworkBuffer>(
            output_data_vec[n]);
    //get confidence result
    int size = result_tensor->GetSize() / sizeof(float);
    **output_buffer_size += size;
  }

  if (**output_buffer_size <= 0 || **output_buffer_size >= 134217728) { // 134217728 = 512M/4
    throw "ai_model_manager Process failed, output buffer size is out of range:1~134217728!";
  }

  *output_buffer = new (nothrow) float[**output_buffer_size];
  for (int n = 0; n < output_data_vec.size(); ++n) {
    std::shared_ptr<hiai::AINeuralNetworkBuffer> result_tensor =
        std::static_pointer_cast<hiai::AINeuralNetworkBuffer>(
            output_data_vec[n]);
    //get confidence result
    int size = result_tensor->GetSize();

    int memcpy_result = memcpy_s(*output_buffer, size,
                                 result_tensor->GetBuffer(), size);
    if (memcpy_result != EOK) { // check memcpy_s result
      ASC_LOG_ERROR(
          "Fail to copy inference result data to output buffer, memcpy_s result:%d",
          memcpy_result);
      throw "Fail to copy inference result data to output buffer!";
    }
  }

  ASC_LOG_INFO("end ai model inference, model path:%s!", model_path_.c_str());
}

void ModelInference::Inference(const string &image_path, ImageType image_type,
                               float** output_buffer,
                               int** output_buffer_size) {

}

}
}
