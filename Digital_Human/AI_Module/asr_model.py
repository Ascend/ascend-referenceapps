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
import torch
import time
from multiprocessing import Process, Queue
from torch.autograd import Variable

sys.path.append("/home/digital_human/AI_DIGITAL_HUMAN/AI_Module")
sys.path.append("/home/digital_human/asr/src")

from base_model import AI_Model
from main import start_asr_inference

""" ASR model takes in .wav format from input queue, processes it, and put output text to the output queue """
class ASR_Model(AI_Model):
    def __init__(self, audio_in_queue, audio_out_queue):
        super().__init__(audio_in_queue, audio_out_queue)
        
        self.asr_inference_proc = Process(
            target=start_asr_inference, args=(self.input_queue, self.output_queue)
        )

    def get_data(self, audio_in_queue, input_data):
        audio_in_queue.put(input_data)

    def run(self):
        self.asr_inference_proc.start()  # Run ASR inference
