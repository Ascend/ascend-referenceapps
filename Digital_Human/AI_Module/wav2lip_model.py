# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

import os
import sys

from multiprocessing import Process, Queue

sys.path.append("/home/digital_human/AI_DIGITAL_HUMAN/AI_Module")
sys.path.append("/home/digital_human/wav2lip_alg")

from base_model import AI_Model
from wav2lip import Wav2lip

""" Wav2lip model takes in audio from input queue, processes it, and put output images to the output queue """

class WAV2LIP_Model(AI_Model):
    def __init__(self, wav2lip_in_queue, wav2lip_out_queue):
        super().__init__(wav2lip_in_queue, wav2lip_out_queue)
        wav2lip = Wav2lip("/home/digital_human/wav2lip_alg/config.yaml")

        self.wav2lip_inference_proc = Process(
            target=wav2lip.run, args=(self.input_queue, self.output_queue)
        )

    def get_data(self, wav2lip_in_queue, input_data):
        wav2lip_in_queue.put(input_data)

    def run(self):
        self.wav2lip_inference_proc.start()  # Run wav2lip inference
