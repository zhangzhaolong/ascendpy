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

#ifndef MODEL_INFERENCE_H_
#define MODEL_INFERENCE_H_

#include <string>
#include <unistd.h>
#include <vector>
#include "toolchain/slog.h"
#include "hiaiengine/ai_model_manager.h"

namespace ascend {
namespace modelinference {

// used for record info-level log info to dlog
#ifndef ASC_LOG_INFO
#define ASC_LOG_INFO(fmt, ...) \
  dlog_info(ASCENDDK, "[%s:%d] " fmt "\n", __FILE__, __LINE__, \
            ##__VA_ARGS__)
#endif

// used for record warning-level log info to dlog
#ifndef ASC_LOG_WARN
#define ASC_LOG_WARN(fmt, ...) \
  dlog_warn(ASCENDDK, "[%s:%d] " fmt "\n", __FILE__,  __LINE__, \
            ##__VA_ARGS__)
#endif

// used for record error-level log info to dlog
#ifndef ASC_LOG_ERROR
#define ASC_LOG_ERROR(fmt, ...) \
  dlog_error(ASCENDDK, "[%s:%d] " fmt "\n", __FILE__,  __LINE__, \
             ##__VA_ARGS__)
#endif

enum ImageType {
  kJpg = 0, // jpg type image
  kPng, // png type image
  kInvalidType // invalid type image
};

class ModelInference {
 public:
  /**
   * @brief   constructor
   */
  ModelInference();

  /**
   * @brief  initialize ai model parameters
   * @param [in] model_path:  the model path
   * @return true:success; false: failed
   */
  void Init(const std::string &model_path);

  /**
   * @brief  initialize ai model parameters
   * @param [in] model_path:  the model path
   * @return true:success; false: failed
   */
  void Init(const std::string &model_path, int iamge_width, int iamge_height);

  /**
   * @brief: inference ai model
   * @param [in] input_buffer: input data buffer
   * @param [in] input_buffer_size: input data buffer size
   * @param [in] output_buffer: output data buffer
   * @param [in] output_buffer_size: output data buffer size
   * @return true:success; false: failed
   */
  void Inference(const char* input_buffer, int input_buffer_size,
                 float** output_buffer, int** output_buffer_size);

  /**
   * @brief: inference ai model
   * @param [in] image_path: the image path used for ai model inference
   * @param [in] image_type: image type
   * @param [in] output_buffer: output data buffer
   * @param [in] output_buffer_size: output data buffer size
   * @return true:success; false: failed
   */
  void Inference(const std::string &image_path, ImageType image_type,
                 float** output_buffer, int** output_buffer_size);

 private:
  /**
   * @brief: verify file path is exist and have read permission
   * @param [in] model_path: model path
   * @return true:verify success; false: verify failed
   */
  bool VerifyFilePath(const std::string &model_path);

  std::string model_path_; // model path

  std::string image_path_; // image path

  int image_width_; // image width

  int image_height_; // image height

  // used for AI model manage
  std::shared_ptr<hiai::AIModelManager> ai_model_manager_;
};

}
}

#endif /* MODEL_INFERENCE_H_ */
