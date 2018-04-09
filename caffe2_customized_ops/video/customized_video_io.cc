/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */
 /**
  * based on:
  * Copyright (c) 2016-present, Facebook, Inc.
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  *     http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */

#include "caffe2/video/customized_video_io.h"
#include <random>
#include <string>
#include "caffe2/core/logging.h"
#include "caffe2/video/customized_video_decoder.h"

namespace caffe2 {

void ImageChannelToBuffer(const cv::Mat* img, float* buffer, int c) {
  int idx = 0;
  for (int h = 0; h < img->rows; ++h) {
    for (int w = 0; w < img->cols; ++w) {
      buffer[idx++] = static_cast<float>(img->at<cv::Vec3b>(h, w)[c]);
    }
  }
}

void ImageDataToBuffer(
    unsigned char* data_buffer,
    int height,
    int width,
    float* buffer,
    int c) {
  int idx = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      buffer[idx++] =
          static_cast<float>(data_buffer[h * width * 3 + w * 3 + c]);
    }
  }
}

bool ReadClipFromFrames(
    std::string img_dir,
    const int start_frm,
    std::string im_extension,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer) {
  char fn_im[512];
  cv::Mat img, img_origin;
  buffer = nullptr;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;

  int end_frm = start_frm + length * sampling_rate;
  for (int i = start_frm; i < end_frm; i += sampling_rate) {
    snprintf(fn_im, 512, "%s/%06d%s", img_dir.c_str(), i, im_extension.c_str());
    if (height > 0 && width > 0) {
      img_origin = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
      if (!img_origin.data) {
        LOG(ERROR) << "Could not open or find file " << fn_im;
        if (buffer != nullptr) {
          delete[] buffer;
        }
        return false;
      }
      cv::resize(img_origin, img, cv::Size(width, height));
      img_origin.release();
    } else {
      img = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
      if (!img.data) {
        LOG(ERROR) << "Could not open or find file " << fn_im;
        if (buffer != nullptr) {
          delete[] buffer;
        }
        return false;
      }
    }

    // If this is the first frame, allocate memory for the buffer
    if (i == start_frm) {
      image_size = img.rows * img.cols;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new float[data_size];
    }

    for (int c = 0; c < 3; c++) {
      ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
    }
    offset += image_size;
  }
  CAFFE_ENFORCE(offset == channel_size, "Wrong offset size");
  return true;
}

int GetNumberOfFrames(std::string filename) {
  cv::VideoCapture cap;
  cap.open(filename);
  if (!cap.isOpened()) {
    LOG(ERROR) << "Cannot open " << filename;
    return 0;
  }
  int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
  cap.release();
  return num_of_frames;
}

double GetVideoFPS(std::string filename) {
  cv::VideoCapture cap;
  cap.open(filename);
  if (!cap.isOpened()) {
    LOG(ERROR) << "Cannot open " << filename;
    return 0;
  }
  double fps = cap.get(CV_CAP_PROP_FPS);
  cap.release();
  return fps;
}

void GetVideoMeta(std::string filename, int& number_of_frames, double& fps) {
  cv::VideoCapture cap;
  cap.open(filename);
  if (cap.isOpened()) {
    number_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    fps = cap.get(CV_CAP_PROP_FPS);
    cap.release();
  } else {
    LOG(ERROR) << "Cannot open " << filename;
    number_of_frames = -1;
    fps = 0;
  }
}

bool ReadClipFromVideoLazzy(
    std::string filename,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer) {
  cv::VideoCapture cap;
  cv::Mat img, img_origin;
  buffer = nullptr;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;
  int end_frm = 0;

  cap.open(filename);
  if (!cap.isOpened()) {
    LOG(ERROR) << "Cannot open " << filename;
    return false;
  }

  int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
  if (num_of_frames < length * sampling_rate) {
    LOG(INFO) << filename << " does not have enough frames; having "
              << num_of_frames;
    return false;
  }

  CAFFE_ENFORCE_GE(start_frm, 0, "start frame must be greater or equal to 0");

  if (start_frm) {
    cap.set(CV_CAP_PROP_POS_FRAMES, start_frm);
  }
  end_frm = start_frm + length * sampling_rate;
  CAFFE_ENFORCE_LE(
      end_frm,
      num_of_frames,
      "end frame must be less or equal to num of frames");

  for (int i = start_frm; i < end_frm; i += sampling_rate) {
    if (sampling_rate > 1) {
      cap.set(CV_CAP_PROP_POS_FRAMES, i);
    }
    if (height > 0 && width > 0) {
      cap.read(img_origin);
      if (!img_origin.data) {
        LOG(INFO) << filename << " has no data at frame " << i;
        if (buffer != nullptr) {
          delete[] buffer;
        }
        return false;
      }
      cv::resize(img_origin, img, cv::Size(width, height));
    } else {
      cap.read(img);
      if (!img.data) {
        LOG(ERROR) << "Could not open or find file " << filename;
        if (buffer != nullptr) {
          delete[] buffer;
        }
        return false;
      }
    }

    // If this is the fisrt frame, allocate memory for the buffer
    if (i == start_frm) {
      image_size = img.rows * img.cols;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new float[data_size];
    }

    for (int c = 0; c < 3; c++) {
      ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
    }

    offset += image_size;
  }

  CAFFE_ENFORCE(offset == channel_size, "wrong offset size");
  cap.release();
  return true;
}

bool ReadClipFromVideoSequential(
    std::string filename,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer) {
  cv::VideoCapture cap;
  cv::Mat img, img_origin;
  buffer = nullptr;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;

  cap.open(filename);
  if (!cap.isOpened()) {
    LOG(ERROR) << "Cannot open " << filename;
    return false;
  }

  int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
  if (num_of_frames < length * sampling_rate) {
    LOG(INFO) << filename << " does not have enough frames; having "
              << num_of_frames;
    return false;
  }

  CAFFE_ENFORCE_GE(start_frm, 0, "start frame must be greater or equal to 0");

  // Instead of random access, do sequentically access (avoid key-frame issue)
  // This will keep start_frm frames
  int sequential_counter = 0;
  while (sequential_counter < start_frm) {
    cap.read(img_origin);
    sequential_counter++;
  }

  int end_frm = start_frm + length * sampling_rate;
  CAFFE_ENFORCE_LE(
      end_frm,
      num_of_frames,
      "end frame must be less or equal to num of frames");

  for (int i = start_frm; i < end_frm; i++) {
    if (sampling_rate > 1) {
      // If sampling_rate > 1, purposely keep some frames
      if ((i - start_frm) % sampling_rate != 0) {
        cap.read(img_origin);
        continue;
      }
    }
    if (height > 0 && width > 0) {
      cap.read(img_origin);
      if (!img_origin.data) {
        LOG(INFO) << filename << " has no data at frame " << i;
        if (buffer != nullptr) {
          delete[] buffer;
        }
        return false;
      }
      cv::resize(img_origin, img, cv::Size(width, height));
    } else {
      cap.read(img);
      if (!img.data) {
        LOG(ERROR) << "Could not open or find file " << filename;
        if (buffer != nullptr) {
          delete[] buffer;
        }
        return false;
      }
    }

    // If this is the first frame, then we allocate memory for the buffer
    if (i == start_frm) {
      image_size = img.rows * img.cols;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new float[data_size];
    }

    for (int c = 0; c < 3; c++) {
      ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
    }

    offset += image_size;
  }
  CAFFE_ENFORCE(offset == channel_size, "wrong offset size");
  cap.release();

  return true;
}

bool ReadClipFromVideo(
    std::string filename,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer) {
  bool read_status = ReadClipFromVideoLazzy(
      filename, start_frm, length, height, width, sampling_rate, buffer);
  if (!read_status) {
    read_status = ReadClipFromVideoSequential(
        filename, start_frm, length, height, width, sampling_rate, buffer);
  }
  return read_status;
}

bool DecodeClipFromVideoFile(
    std::string filename,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer) {
  Params params;
  std::vector<std::unique_ptr<DecodedFrame>> sampledFrames;
  CustomVideoDecoder decoder;

  params.outputHeight_ = height ? height : -1;
  params.outputWidth_ = width ? width : -1;
  params.maximumOutputFrames_ = MAX_DECODING_FRAMES;

  // decode all frames with defaul sampling rate
  decoder.decodeFile(filename, params, sampledFrames);

  buffer = nullptr;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;

  int end_frm = start_frm + length * sampling_rate;
  for (int i = start_frm; i < end_frm; i += sampling_rate) {
    if (i == start_frm) {
      image_size = sampledFrames[i]->height_ * sampledFrames[i]->width_;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new float[data_size];
    }

    for (int c = 0; c < 3; c++) {
      ImageDataToBuffer(
          (unsigned char*)sampledFrames[i]->data_.get(),
          sampledFrames[i]->height_,
          sampledFrames[i]->width_,
          buffer + c * channel_size + offset,
          c);
    }
    offset += image_size;
  }
  CAFFE_ENFORCE(offset == channel_size, "Wrong offset size");

  // free the sampledFrames
  for (int i = 0; i < sampledFrames.size(); i++) {
    DecodedFrame* p = sampledFrames[i].release();
    delete p;
  }
  sampledFrames.clear();

  return true;
}

bool DecodeClipFromMemoryBuffer(
    const char* video_buffer,
    const int size,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer,
    std::mt19937* randgen) {
  Params params;
  std::vector<std::unique_ptr<DecodedFrame>> sampledFrames;
  CustomVideoDecoder decoder;

  params.outputHeight_ = height ? height : -1;
  params.outputWidth_ = width ? width : -1;
  params.maximumOutputFrames_ = MAX_DECODING_FRAMES;

  bool isTemporalJitter = (start_frm < 0);

  // decoder.decodeMemory(
  //     video_buffer,
  //     size,
  //     params,
  //     sampledFrames,
  //     length * sampling_rate,
  //     !isTemporalJitter);
  //
  // if (sampledFrames.size() < length * sampling_rate) {
  //   /* selective decoding failed. Decode all frames. */
  //   decoder.decodeMemory(video_buffer, size, params, sampledFrames);
  // }
  decoder.decodeMemory(video_buffer, size, params, sampledFrames);

  buffer = nullptr;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;

  int use_start_frm = start_frm;
  if (start_frm < 0) { // perform temporal jittering
    if ((int)(sampledFrames.size() - length * sampling_rate) > 0) {
      use_start_frm = std::uniform_int_distribution<>(
          0, (int)(sampledFrames.size() - length * sampling_rate))(*randgen);
    } else {
      use_start_frm = 0;
    }
  }

  if (sampledFrames.size() == 0) {
    LOG(ERROR) << "This video is empty.";
    buffer = nullptr;
    return true;
  }

  for (int idx = 0; idx < length; idx ++){
    int i = use_start_frm + idx * sampling_rate;
    // TODO{km}: consider cylindric sampling
    i = i % (int)(sampledFrames.size());  // periodic sampling
    if (idx == 0) {
      image_size = sampledFrames[i]->height_ * sampledFrames[i]->width_;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new float[data_size];
    }

    for (int c = 0; c < 3; c++) {
      ImageDataToBuffer(
          (unsigned char*)sampledFrames[i]->data_.get(),
          sampledFrames[i]->height_,
          sampledFrames[i]->width_,
          buffer + c * channel_size + offset,
          c);
    }
    offset += image_size;
  }
  CAFFE_ENFORCE(offset == channel_size, "Wrong offset size");

  // free the sampledFrames
  for (int i = 0; i < sampledFrames.size(); i++) {
    DecodedFrame* p = sampledFrames[i].release();
    delete p;
  }
  sampledFrames.clear();

  return true;
}


// ----------------------------------------------------------------
// customized functions follow
// ----------------------------------------------------------------

void ClipTransformFlex(
    const float* clip_data,
    const int channels,
    const int length,
    const int height,
    const int width,
    const int h_crop,
    const int w_crop,
    const bool mirror,
    float mean,
    float std,
    float* transformed_clip,
    std::mt19937* randgen,
    std::bernoulli_distribution* mirror_this_clip,
    const bool use_center_crop,
    const bool use_bgr,
    const int spatial_pos
  ) {
  int h_off = 0;
  int w_off = 0;

  assert(height >= h_crop);
  assert(width >= w_crop);

  if (use_center_crop) {
    h_off = (height - h_crop) / 2;
    w_off = (width - w_crop) / 2;
    if (spatial_pos >= 0)
    {
      int now_pos = spatial_pos % 3;
      if (h_off > 0) h_off = h_off * now_pos;
      else w_off = w_off * now_pos;
    }

  } else {
    h_off = std::uniform_int_distribution<>(0, height - h_crop)(*randgen);
    w_off = std::uniform_int_distribution<>(0, width - w_crop)(*randgen);
  }

  float inv_std = 1.f / std;
  int top_index, data_index;
  bool mirror_me = mirror && (*mirror_this_clip)(*randgen);
  if (spatial_pos >= 0)
  {
    mirror_me = int(spatial_pos / 3);
  }

  for (int c = 0; c < channels; ++c) {
    for (int l = 0; l < length; ++l) {
      for (int h = 0; h < h_crop; ++h) {
        for (int w = 0; w < w_crop; ++w) {
          if (!use_bgr) { // rgb as is
            data_index =
                ((c * length + l) * height + h_off + h) * width + w_off + w;
          } else {
            data_index =
                (((channels - c - 1) * length + l) * height + h_off + h) * width
                + w_off + w;
          }
          if (mirror_me) {
            top_index = ((c * length + l) * h_crop + h) * w_crop +
                (w_crop - 1 - w);
          } else {
            top_index = ((c * length + l) * h_crop + h) * w_crop + w;
          }
          transformed_clip[top_index] =
              (clip_data[data_index] - mean) * inv_std;
        }
      }
    }
  }
}

void ScaleTransform(
    const float* clip_data,
    const int channels,
    const int length,
    const int height,
    const int width,
    const int max_size,
    const int min_size,
    float*& buffer,
    std::mt19937* randgen,
    int & new_height,
    int & new_width)
    {
      int side_length;

      if (min_size == max_size)
      {
        side_length = min_size;
      }
      else
      {
        side_length =
          std::uniform_int_distribution<>(min_size, max_size)(*randgen);
      }
      new_height = height;
      new_width  = width;

      float ratio = 1;

      buffer = nullptr;

      if (height > width)
      {
        ratio = (float)side_length / (float)width;
      }
      else
      {
        ratio = (float)side_length / (float)height;
      }
      new_height = (int)((float)height * ratio);
      new_width = (int)((float)width * ratio);

      cv::Mat img(cv::Size(new_width, new_height), CV_8UC3);
      cv::Mat img_origin(cv::Size(width, height), CV_8UC3);

      int image_size = new_height * new_width;
      int channel_size = image_size * length;
      int data_size = channel_size * 3;
      buffer = new float[data_size];
      int offset = 0;

      for (int l = 0; l < length; ++l)
      {
        for (int c = 0; c < 3; ++c)
        {
          for (int h = 0; h < height; ++h)
          {
            for (int w = 0; w < width; ++w)
            {
              int data_index = ((c * length + l) * height  + h) * width  + w;
              float tnum = clip_data[data_index];
              img_origin.at<cv::Vec3b>(h, w)[c] = (uchar)(tnum);
              // img_origin.at<cv::Vec3b>(h, w)[2 - c] = (uchar)(tnum);
            } // w
          } // h
        } // c
        cv::resize(img_origin, img, cv::Size(new_width, new_height));

        for (int c = 0; c < 3; c++) {
          ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
        } // c
        offset += image_size;
      } // l
      CAFFE_ENFORCE(offset == channel_size, "Wrong offset size");
      img_origin.release();
      img.release();
    } // ScaleTransform


// for reading file from lmdb
bool DecodeClipFromVideoFileFlex(
    std::string filename,
    const int start_frm,
    const int length,
    int & height,
    int & width,
    const int sampling_rate,
    float*& buffer,
    std::mt19937* randgen,
    const int sample_times
  ) {
  Params params;
  std::vector<std::unique_ptr<DecodedFrame>> sampledFrames;
  CustomVideoDecoder decoder;

  params.outputHeight_ = -1;
  params.outputWidth_ = -1;
  params.maximumOutputFrames_ = MAX_DECODING_FRAMES;

  // decode all frames with defaul sampling rate
  decoder.decodeFile(filename, params, sampledFrames);

  buffer = nullptr;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;
  CAFFE_ENFORCE_LT(1, sampledFrames.size(), "video cannot be empty");

  int use_start_frm = start_frm;
  if (start_frm < 0) { // perform temporal jittering
    if ((int)(sampledFrames.size() - length * sampling_rate) > 0) {
      use_start_frm = std::uniform_int_distribution<>(
          0, (int)(sampledFrames.size() - length * sampling_rate))(*randgen);
    } else { use_start_frm = 0; }
  }
  else
  {
    int num_of_frames = (int)(sampledFrames.size());
    float frame_gaps = (float)(num_of_frames) / (float)(sample_times);
    use_start_frm = ((int)(frame_gaps * start_frm)) % num_of_frames;
  }


  height = (int)sampledFrames[0]->height_;
  width  = (int)sampledFrames[0]->width_;

  for (int idx = 0; idx < length; idx ++){
    int i = use_start_frm + idx * sampling_rate;
    i = i % (int)(sampledFrames.size());
    if (idx == 0) {
      image_size = sampledFrames[i]->height_ * sampledFrames[i]->width_;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new float[data_size];
    }

    for (int c = 0; c < 3; c++) {
      ImageDataToBuffer(
          (unsigned char*)sampledFrames[i]->data_.get(),
          sampledFrames[i]->height_,
          sampledFrames[i]->width_,
          buffer + c * channel_size + offset,
          c);
    }
    offset += image_size;
  }
  CAFFE_ENFORCE(offset == channel_size, "Wrong offset size");

  // free the sampledFrames
  for (int i = 0; i < sampledFrames.size(); i++) {
    DecodedFrame* p = sampledFrames[i].release();
    delete p;
  }
  sampledFrames.clear();

  return true;
}


// for reading file from memory buffer
bool DecodeClipFromMemoryBufferFlex(
    const char* video_buffer,
    const int size,
    const int start_frm,
    const int length,
    int & height,
    int & width,
    const int sampling_rate,
    float*& buffer,
    std::mt19937* randgen) {
  Params params;
  std::vector<std::unique_ptr<DecodedFrame>> sampledFrames;
  CustomVideoDecoder decoder;

  params.outputHeight_ = -1;
  params.outputWidth_ =  -1;
  params.maximumOutputFrames_ = MAX_DECODING_FRAMES;

  // ----------- usable with selective decoding
  // bool isTemporalJitter = (start_frm < 0);
  // decoder.decodeMemory(
  //     video_buffer,
  //     size,
  //     params,
  //     sampledFrames,
  //     length * sampling_rate,
  //     !isTemporalJitter);
  //
  // if (sampledFrames.size() < length * sampling_rate) {
  //   /* selective decoding failed. Decode all frames. */
  //   decoder.decodeMemory(video_buffer, size, params, sampledFrames);
  // }
  // ----------- usable with selective decoding

  decoder.decodeMemory(video_buffer, size, params, sampledFrames);

  buffer = nullptr;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;

  int use_start_frm = start_frm;
  if (start_frm < 0) { // perform temporal jittering
    if ((int)(sampledFrames.size() - length * sampling_rate) > 0) {
      use_start_frm = std::uniform_int_distribution<>(
          0, (int)(sampledFrames.size() - length * sampling_rate))(*randgen);
    } else { use_start_frm = 0; }
  }

  if (sampledFrames.size() == 0) {
    LOG(ERROR) << "This video is empty.";
    buffer = nullptr;
    return true;
  }

  height = (int)sampledFrames[0]->height_ ;
  width  = (int)sampledFrames[0]->width_;

  for (int idx = 0; idx < length; idx ++){
    int i = use_start_frm + idx * sampling_rate;
    // TODO{km}: consider cylindric sampling
    i = i % (int)(sampledFrames.size());
    if (idx == 0) {
      image_size = sampledFrames[i]->height_ * sampledFrames[i]->width_;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new float[data_size];
    }

    for (int c = 0; c < 3; c++) {
      ImageDataToBuffer(
          (unsigned char*)sampledFrames[i]->data_.get(),
          sampledFrames[i]->height_,
          sampledFrames[i]->width_,
          buffer + c * channel_size + offset,
          c);
    }
    offset += image_size;
  }
  CAFFE_ENFORCE(offset == channel_size, "Wrong offset size");

  // free the sampledFrames
  for (int i = 0; i < sampledFrames.size(); i++) {
    DecodedFrame* p = sampledFrames[i].release();
    delete p;
  }
  sampledFrames.clear();

  return true;
}


} // caffe2 namespace
