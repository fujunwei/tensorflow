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
"""convTranspose2d for tflite."""

import argparse

import numpy as np
import tflite_runtime.interpreter as tflite

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-m',
      '--model_file',
      default='./pool2d.tflite',
      help='.tflite model to be executed')

  args = parser.parse_args()

  interpreter = tflite.Interpreter(
      model_path=args.model_file,
      experimental_delegates=None,
      num_threads=1)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  print("input_details:")
  print(input_details)
  output_details = interpreter.get_output_details()
  print("output_details:")
  print(output_details)

  # [1, 3, 3, 1]
  input_data = np.array([
    22.975555502750634,
            78.15438048012338,
            9.68611138116071,
            51.29803808129347,
            32.19308601456918,
            87.65037289600019,
            87.25082191311348,
            39.49793996935087,
            80.09963591169489,
            10.220142557736978,
            52.60270021646585,
            1.4128639882603933,
            11.954064466077474,
            85.0007506374375,
            64.7837446465813,
            88.03128735720126,
            11.333851214909307,
            70.61659435728073,
            84.90442561999888,
            79.06688041781518,
            7.328724951604215,
            35.97796581186121,
            10.17730631094398,
            1.4140757517112412,
            78.10038172113374,
            91.59549689157087,
            65.64701225681809,
            55.14215004436653,
            18.432438840756184,
            49.34624267439973,
            15.648024969290454,
            68.02723372727797,
            20.342549040418124,
            26.72794900604616,
            64.87446829774323,
            46.56714896227794,
            79.57832937136276,
            4.338463748959498,
            38.18383968382213,
            45.253981324455175,
            80.9717996657439,
            67.58124910163149,
            6.026499585657263,
            29.77881349289366,
            58.58993337807239,
            2.2384984647495054,
            14.505490166700486,
            68.72449589246624,
            76.45657404642184,
            23.53263275794233
  ], dtype=np.float32)
  input_data = np.reshape(input_data, (1, 2, 5, 5))
  print('input shape: {}'.format(input_data.shape))

  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  print('output data: {}'.format(output_data))
