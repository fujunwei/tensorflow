# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

import argparse
import time

import numpy as np
import tflite_runtime.interpreter as tflite

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-m',
      '--model_file',
      default='/home/junwei/workspace/webnn/tflite_python/ops_test/conv_transpose2d.tflite',
      help='.tflite model to be executed')

  args = parser.parse_args()

  interpreter = tflite.Interpreter(
      model_path=args.model_file,
      experimental_delegates=None,
      num_threads=1)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  # floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  # add N dim
  input_data = np.array([[[[0.5872158177067033, 0.6077792328258038], 
                           [0.01728916618181975, 0.26146076483771563]]]], dtype=np.float32)
  print('input shape: {}'.format(input_data.shape))

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  print('output data: {}'.format(output_data))
