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
   
#ifndef CUSTOMIZED_VIDEO_INPUT_OP_H_
#define CUSTOMIZED_VIDEO_INPUT_OP_H_

#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>


#include <opencv2/opencv.hpp>

#include "caffe2/core/db.h"
#include "caffe2/core/logging.h"
#include "caffe2/operators/prefetch_op.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/thread_pool.h"
// #include "caffe2/video/video_io.h"
#include "caffe2/video/customized_video_io.h"

namespace caffe2 {

template <class Context>
class CustomizedVideoInputOp final : public PrefetchOperator<Context> {
 public:
  using OperatorBase::OutputSize;
  using PrefetchOperator<Context>::context_;
  using PrefetchOperator<Context>::prefetch_thread_;
  explicit CustomizedVideoInputOp(
    const OperatorDef& operator_def, Workspace* ws);
  ~CustomizedVideoInputOp() {
    PrefetchOperator<Context>::Finalize();
  }

  // override methods
  bool Prefetch() override;
  bool CopyPrefetched() override;

 private:
  bool GetClipAndLabelFromDBValue(
      const std::string& value,
      float*& buffer,
      int* label_data,
      std::mt19937* randgen,
      int & height,
      int & width);

  void DecodeAndTransform(
      const std::string value,
      float* clip_data,
      int* label_data,
      const int crop_size,
      const bool mirror,
      const float mean,
      const float std,
      std::mt19937* randgen,
      std::bernoulli_distribution* mirror_this_clip,
      int* height_out,
      int* width_out);

  const db::DBReader* reader_;
  CPUContext cpu_context_;
  TensorCPU prefetched_clip_;
  TensorCPU prefetched_label_;
  Tensor<Context> prefetched_clip_on_device_;
  Tensor<Context> prefetched_label_on_device_;
  int batch_size_;
  float mean_;
  float std_;
  int crop_;
  int scale_h_;
  int scale_w_;
  int length_;
  int sampling_rate_;
  bool mirror_;
  bool temporal_jitter_;
  bool use_image_;
  bool multiple_label_;
  int num_of_labels_;
  bool use_local_file_;
  bool is_test_;
  std::string im_extension_;

  // thread pool for parse + decode
  int num_decode_threads_;

  // extra attributes follow
  int use_bgr_;

  int min_size_;
  int max_size_;
  bool use_scale_augmentaiton_;

  int sample_times_;
  int use_multi_crop_;

  std::shared_ptr<TaskThreadPool> thread_pool_;
};

template <class Context>
CustomizedVideoInputOp<Context>::CustomizedVideoInputOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : PrefetchOperator<Context>(operator_def, ws),
      reader_(nullptr),
      batch_size_(
          OperatorBase::template GetSingleArgument<int>("batch_size", 0)),
      mean_(OperatorBase::template GetSingleArgument<float>("mean", 0.)),
      std_(OperatorBase::template GetSingleArgument<float>("std", 1.)),
      crop_(OperatorBase::template GetSingleArgument<int>("crop", -1)),
      scale_h_(OperatorBase::template GetSingleArgument<int>("height", 0)),
      scale_w_(OperatorBase::template GetSingleArgument<int>("width", 0)),
      length_(OperatorBase::template GetSingleArgument<int>("length", 0)),
      sampling_rate_(
          OperatorBase::template GetSingleArgument<int>("sampling_rate", 1)),
      mirror_(OperatorBase::template GetSingleArgument<int>("mirror", 0)),
      temporal_jitter_(
          OperatorBase::template GetSingleArgument<int>("temporal_jitter", 1)),
      use_image_(OperatorBase::template GetSingleArgument<int>("use_image", 0)),
      multiple_label_(
          OperatorBase::template GetSingleArgument<int>("multiple_label", 0)),
      num_of_labels_(
          OperatorBase::template GetSingleArgument<int>("num_of_labels", 0)),
      use_local_file_(
          OperatorBase::template GetSingleArgument<int>("use_local_file", 0)),
      is_test_(OperatorBase::template GetSingleArgument<int>("is_test", 0)),
      im_extension_(
          OperatorBase::template GetSingleArgument<string>("im_extension", "")),
      num_decode_threads_(
          OperatorBase::template GetSingleArgument<int>("decode_threads", 4)),
      use_bgr_(OperatorBase::template GetSingleArgument<int>("use_bgr", 0)),
      min_size_(OperatorBase::template GetSingleArgument<int>("min_size", 256)),
      max_size_(OperatorBase::template GetSingleArgument<int>("max_size", 480)),
      use_scale_augmentaiton_(
          OperatorBase::template GetSingleArgument<int>(
            "use_scale_augmentaiton", 0)),
      sample_times_(
          OperatorBase::template GetSingleArgument<int>("sample_times", 10)),
      use_multi_crop_(
          OperatorBase::template GetSingleArgument<int>("use_multi_crop", 0)),

      thread_pool_(new TaskThreadPool(num_decode_threads_)) {
  CAFFE_ENFORCE_GT(batch_size_, 0, "Batch size should be nonnegative.");
  // CAFFE_ENFORCE_GE(scale_h_, 0, "Must provide the scale value.");
  // CAFFE_ENFORCE_GE(scale_w_, 0, "Must provide the cropping value.");
  CAFFE_ENFORCE_GT(length_, 0, "Must provide the clip length value.");
  // CAFFE_ENFORCE_GT(crop_, 0, "Must provide the cropping value.");
  // CAFFE_ENFORCE_GE(
  //     scale_h_,
  //     crop_,
  //     "The scaled height must be no smaller than the crop value.");
  // CAFFE_ENFORCE_GE(
  //     scale_w_,
  //     crop_,
  //     "The scaled width must be no smaller than the crop value.");

  if (multiple_label_) {
    CAFFE_ENFORCE_GT(
        num_of_labels_,
        0,
        "Number of labels must be set for using multiple label output.");
  }
  if (crop_ <= 0){  // not cropping
    CAFFE_ENFORCE_EQ(
        is_test_,
        1,
        "Cannot use spatial uncrop at training.");
  }

  // Always need a dbreader, even when using local video files
  CAFFE_ENFORCE_GT(
      operator_def.input_size(), 0, "Need to have a DBReader blob input");

  LOG(INFO) << "Creating a clip input op with the following setting: ";
  LOG(INFO) << "    Using " << num_decode_threads_ << " CPU threads;";
  if (temporal_jitter_) {
    LOG(INFO) << "  Using temporal jittering;";
  }
  LOG(INFO) << "    Outputting in batches of " << batch_size_ << " clips;";
  LOG(INFO) << "    Scaling image to " << scale_h_ << "x" << scale_w_;

  LOG(INFO) << "    Cropping video frame to " << crop_
            << (mirror_ ? " with " : " without ") << "random mirroring;";
  LOG(INFO) << "    Using " << (is_test_ ? "center" : "random") << " crop";
  LOG(INFO) << "    Using a clip of " << length_ << " frames;";
  LOG(INFO) << "    Using a sampling rate of 1:" << sampling_rate_;
  LOG(INFO) << "    Subtract mean " << mean_ << " and divide by std " << std_
            << ".";

  // extra attributes follow
  LOG(INFO) << "    Scaling image from " << min_size_ << " to " << max_size_;
  LOG(INFO) << "    Using scale augmentaiton?: " << use_scale_augmentaiton_ ;
  LOG(INFO) << "    Using BGR order?: " << use_bgr_ ;
  LOG(INFO) << "    Using sample_times_:" << sample_times_;
  LOG(INFO) << "    Using use_multi_crop_: " << use_multi_crop_ ;


  vector<TIndex> data_shape(5);
  vector<TIndex> label_shape(2);

  data_shape[0] = batch_size_;
  // Assume color videos, will convert to 3 channels, even with black & with
  // input videos
  data_shape[1] = 3;
  data_shape[2] = length_;
  if (crop_ > 0) {
    data_shape[3] = crop_;
    data_shape[4] = crop_;
  } else {  // uncrop
    data_shape[3] = 320;  // rough estimate
    data_shape[4] = 320;  // rough estimate
  }
  prefetched_clip_.Resize(data_shape);

  // If multiple label is used, outout label is a binary vector of length
  // number of labels-dim in indicating which labels present
  if (multiple_label_) {
    label_shape[0] = batch_size_;
    label_shape[1] = num_of_labels_;
    prefetched_label_.Resize(label_shape);
  } else {
    prefetched_label_.Resize(vector<TIndex>(1, batch_size_));
  }
}

template <class Context>
bool CustomizedVideoInputOp<Context>::GetClipAndLabelFromDBValue(
    const string& value,
    float*& buffer,
    int* label_data,
    std::mt19937* randgen,
    int & height,
    int & width
  ) {
  TensorProtos protos;
  CAFFE_ENFORCE(protos.ParseFromString(value));
  const TensorProto& video_proto = protos.protos(0);
  const TensorProto& label_proto = protos.protos(1);

  int start_frm = -1;
  if (!temporal_jitter_) {
    const TensorProto& start_frm_proto = protos.protos(2);
    start_frm = start_frm_proto.int32_data(0);
  }
  // int start_frm = temporal_jitter_ ? -1 : 0;

  // assign labels
  if (!multiple_label_) {
      label_data[0] = label_proto.int32_data(0);
  } else {
    // For multiple label case, output label is a binary vector
    // where presented concepts are makred 1
    memset(label_data, 0, sizeof(int) * num_of_labels_);
    for (int i = 0; i < label_proto.int32_data_size(); i++) {
      label_data[label_proto.int32_data(i)] = 1;
    }
  }

  if (use_local_file_) {
    CAFFE_ENFORCE_EQ(
        video_proto.data_type(),
        TensorProto::STRING,
        "Database with a file_list is expected to be string data");
  }

  if (video_proto.data_type() == TensorProto::STRING) {
    const string& encoded_video_str = video_proto.string_data(0);
    int encoded_size = encoded_video_str.size();
    if (!use_local_file_) {
      DecodeClipFromMemoryBufferFlex(
          const_cast<char*>(encoded_video_str.data()),
          encoded_size,
          start_frm,
          length_,
          height,
          width,
          sampling_rate_,
          buffer,
          randgen);
    } else { // use local file
      // encoded string contains an absolute path to a local file or folder
      std::string filename = encoded_video_str;
      if (use_image_) {
        LOG(FATAL) << "Branch not implemented.";
        /* CAFFE_ENFORCE(
          !temporal_jitter_,
          "Temporal jittering is not suported for image sequence input"
        );
        CHECK(ReadClipFromFrames(
            filename,
            start_frm,
            im_extension_,
            length_,
            scale_h_,
            scale_w_,
            sampling_rate_,
            buffer)); */
      } else {
        // printf("filename: %s\n", filename.c_str());
        CHECK(DecodeClipFromVideoFileFlex(
            filename,
            start_frm,
            length_,
            height,
            width,
            sampling_rate_,
            buffer,
            randgen,
            sample_times_
          ));
      } // end of else (i.e., use_image_ == False)
    } // end of else (i.e., use_local_file_ == True)
  } else if (video_proto.data_type() == TensorProto::BYTE) {
    LOG(FATAL) << "Branch not implemented.";
    /* DecodeClipFromMemoryBufferFlex(
        video_proto.byte_data().data(),
        video_proto.byte_data().size(),
        start_frm,
        length_,
        height,
        width,
        sampling_rate_,
        buffer,
        randgen); */
  } else {
    LOG(FATAL) << "Unknown video data type.";
  }
  return true;
}

template <class Context>
void CustomizedVideoInputOp<Context>::DecodeAndTransform(
    const std::string value,
    float* clip_data,
    int* label_data,
    const int crop_size,  // -1 is uncrop
    const bool mirror,
    const float mean,
    const float std,
    std::mt19937* randgen,
    std::bernoulli_distribution* mirror_this_clip,
    int* height_out,
    int* width_out
  ) {
  float* buffer = nullptr;

  // Decode the video from memory or read from a local file
  int height_raw = -1;
  int width_raw = -1;
  int height_scaled = -1;
  int width_scaled = -1;
  CHECK(GetClipAndLabelFromDBValue(
    value, buffer, label_data, randgen, height_raw, width_raw)
  );

  if ((height_raw <= 0) || (width_raw <= 0)) return;

  const int num_clips = 1;

  for (int i = 0; i < num_clips; i ++) {
    if (use_scale_augmentaiton_) {
      const int buffer_sample_size = height_raw * width_raw * length_ * 3;
      float* buffer_scaled = nullptr;

      ScaleTransform(
          buffer + buffer_sample_size * i,
          3,
          length_,
          height_raw,
          width_raw,
          max_size_,
          min_size_,
          buffer_scaled,
          randgen,
          height_scaled,
          width_scaled);


      // determine the returned output size
      if (i == 0) {
        if (crop_size > 0) {
          *height_out = crop_size;
          *width_out = crop_size;
        } else {
          *height_out = height_scaled;
          *width_out = width_scaled;

          // avoid extreme size (even if we want "full" image)
          const float ratio_max = 1.6f;
          const float ratio = (height_scaled > width_scaled) ?
            ((float)height_scaled / width_scaled) :
            ((float)width_scaled / height_scaled);

          if (ratio > ratio_max) {
            if (height_scaled > width_scaled) {
                *height_out = (int)(ratio_max * width_scaled);
            } else {
                *width_out = (int)(ratio_max * height_scaled);
            }
            // LOG(INFO) << "Truncate image: "
            //   << "width: " << width_scaled << ", height: " << height_scaled
            //   << " to width: " << *width_out << ", height: " << *height_out;
          } // if ratio > ratio_max
        } // else crop_size > 0
      } // if i

      const int clip_size = *height_out * *width_out * length_ * 3;
      int spatial_pos = -1;
      if (use_multi_crop_ > 0)
      {
        // crop along the longer side
        TensorProtos protos;
        CAFFE_ENFORCE(protos.ParseFromString(value));
        const TensorProto& spatial_pos_proto = protos.protos(3);
        spatial_pos = spatial_pos_proto.int32_data(0);
      }

      ClipTransformFlex(
          buffer_scaled,
          3,
          length_,
          height_scaled,
          width_scaled,
          (*height_out),
          (*width_out),
          mirror,
          mean,
          std,
          clip_data + clip_size * i,
          randgen,
          mirror_this_clip,
          is_test_,
          use_bgr_,
          spatial_pos
        );

      if (buffer_scaled != nullptr)
        delete[] buffer_scaled;
    } else {
      LOG(FATAL) << "We don't recommend using unrestricted input size, "
      << "as it is heavily dependent on dataset preparation.";

      // const int buffer_sample_size = height * width * length_ * 3;
      // it will caused problem is the side < crop_size
      // ClipTransform(
      //     buffer + buffer_sample_size * i,
      //     3,
      //     length_,
      //     height,
      //     width,
      //     crop_size + ,
      //     mirror,
      //     mean,
      //     std,
      //     clip_data + clip_size * i,
      //     randgen,
      //     mirror_this_clip,
      //     is_test_);
    } // else
  } // i
  delete[] buffer;
}

template <class Context>
bool CustomizedVideoInputOp<Context>::Prefetch() {
  // We will get the reader pointer from input.
  // If we use local clips, db will store the list
  reader_ = &OperatorBase::Input<db::DBReader>(0);

  const int channels = 3;

  // Call mutable_data() once to allocate the underlying memory.
  prefetched_clip_.mutable_data<float>();
  prefetched_label_.mutable_data<int>();

  // Prefetching handled with a thread pool of "decode_threads" threads.
  std::mt19937 meta_randgen(time(nullptr));
  std::vector<std::mt19937> randgen_per_thread;
  for (int i = 0; i < num_decode_threads_; ++i) {
    randgen_per_thread.emplace_back(meta_randgen());
  }

  std::bernoulli_distribution mirror_this_clip(0.5);

  const int num_items = batch_size_;

  // ------------ only useful for crop_ <= 0
  std::vector<float*> list_clip_data;
  std::vector<int> list_height_out;
  std::vector<int> list_width_out;
  list_clip_data.resize(num_items);
  list_height_out.resize(num_items);
  list_width_out.resize(num_items);
  const int MAX_IMAGE_SIZE = 500 * 500;
  if (crop_ <= 0) {
    for (int item_id = 0; item_id < num_items; ++item_id) {
      const int num_clips = 1;
      /*
      we have to allocate outside of DecodeAndTransform,
      because DecodeAndTransform does not change the values.
      */
      list_clip_data[item_id] =
        new float[num_clips * MAX_IMAGE_SIZE * length_ * 3];
      list_height_out[item_id] = -1;
      list_width_out[item_id] = -1;
    } // for
  } //if
  // ------------------------

  for (int item_id = 0; item_id < num_items; ++item_id) {
    std::mt19937* randgen = &randgen_per_thread[item_id % num_decode_threads_];

    // get the label data pointer for the item_id -th example
    int* label_data = prefetched_label_.mutable_data<int>() +
        (multiple_label_ ? num_of_labels_ : 1) * item_id;

    // float* clip_data = prefetched_clip_.mutable_data<float>() +
    //   crop_ * crop_ * length_ * channels * item_id;

    std::string key, value;
    // read data
    reader_->Read(&key, &value);
    thread_pool_->runTask(std::bind(
        &CustomizedVideoInputOp<Context>::DecodeAndTransform,
        this,
        std::string(value),
        (crop_ > 0) ?
        (prefetched_clip_.mutable_data<float>() +
          crop_ * crop_ * length_ * channels * item_id) // clip_data
        : (list_clip_data[item_id]), // temp list
        label_data,
        crop_,
        mirror_,
        mean_,
        std_,
        randgen,
        &mirror_this_clip,
        &(list_height_out[item_id]),
        &(list_width_out[item_id])
      ));
  } // for over the batch
  thread_pool_->waitWorkComplete();

  // ------------ only useful for crop_ <= 0
  if (crop_ <= 0) {  // There should be only one item
    if (num_items != 1) {
      LOG(FATAL) << "There should be only one item.";
    }
    if (MAX_IMAGE_SIZE < list_height_out[0] * list_width_out[0]) {
      LOG(FATAL) << "Buffer is too small.";
    }

    // reallocate
    /*
    The network is usually designed for 224x224 input. If the empty image is
    smaller than this size, the network run can crash (e.g., kernel > space)
    */
    const int MIN_SIZE = 224;
    vector<TIndex> data_shape(5);
    data_shape[0] = batch_size_;
    data_shape[1] = 3;
    data_shape[2] = length_;
    data_shape[3] = std::max(list_height_out[0], MIN_SIZE); // for safety
    data_shape[4] = std::max(list_width_out[0], MIN_SIZE); // for safety
    prefetched_clip_.Resize(data_shape);
    prefetched_clip_.mutable_data<float>();
    if (list_height_out[0] < MIN_SIZE || list_width_out[0] < MIN_SIZE) {
      LOG(ERROR) << "Video is too small.";
    }

    // in case of empty video, initialize an all-zero blob
    memset(prefetched_clip_.mutable_data<float>(), 0,
      sizeof(float) * prefetched_clip_.size());
    if (list_clip_data[0] != nullptr
        && list_height_out[0] > 0 && list_width_out[0] > 0) {
      const int num_clips = batch_size_;
      memcpy(
          prefetched_clip_.mutable_data<float>(),
          list_clip_data[0],
          sizeof(float) * num_clips *
          list_height_out[0] * list_width_out[0] * length_ * 3
      );
      delete [] list_clip_data[0];
      list_clip_data[0] = nullptr;
    }
  } // if crop_ <= 0
  // ------------------------

  // If the context is not CPUContext, we will need to do a copy in the
  // prefetch function as well.
  if (!std::is_same<Context, CPUContext>::value) {
    prefetched_clip_on_device_.CopyFrom(prefetched_clip_, &context_);
    prefetched_label_on_device_.CopyFrom(prefetched_label_, &context_);
  }
  return true;
}

template <class Context>
bool CustomizedVideoInputOp<Context>::CopyPrefetched() {
  auto* clip_output = OperatorBase::Output<Tensor<Context>>(0);
  auto* label_output = OperatorBase::Output<Tensor<Context>>(1);
  if (std::is_same<Context, CPUContext>::value) {
    clip_output->CopyFrom(prefetched_clip_, &context_);
    label_output->CopyFrom(prefetched_label_, &context_);
  } else {
    clip_output->CopyFrom(prefetched_clip_on_device_, &context_);
    label_output->CopyFrom(prefetched_label_on_device_, &context_);
  }
  return true;
}

} // namespace caffe2

#endif // CUSTOMIZED_VIDEO_INPUT_OP_H_
