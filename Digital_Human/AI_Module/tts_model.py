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
sys.path.append("/home/digital_human/tts")

from base_model import AI_Model
from streaming_tts_queue import start_tts_inference

""" TTS model takes in text format from input queue, processes it, and put output audio to the output queue """

class TTS_Model(AI_Model):
    def __init__(self, tts_in_queue, tts_out_queue):
        super().__init__(tts_in_queue, tts_out_queue)

        self.tts_inference_proc = Process(
            target=start_tts_inference, args=(self.tts_in_queue, self.tts_out_queue)
        )

    def get_data(self, tts_in_queue, input_data):
        tts_in_queue.put(input_data)

    def run(self):
        self.tts_inference_proc.start()  # Run tts inference
