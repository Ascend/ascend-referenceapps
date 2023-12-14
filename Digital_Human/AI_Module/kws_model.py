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
from multiprocessing import Process, Queue
from torch.autograd import Variable

sys.path.append("/home/digital_human/AI_DIGITAL_HUMAN/AI_Module")
sys.path.append("/home/digital_human/kwx")

from base_model import AI_Model
from om_infer import start_kws_inference

""" KWS model takes in .wav format from input queue, processes it, and put output text to the output queue """

class KWS_Model(AI_Model):
    def __init__(self, input_queue, output_queue):
        super().__init__(input_queue, output_queue)
        
        self.kws_inference_proc = Process(
            target=start_kws_inference, args=(self.input_queue, self.output_queue)
        )

    def get_data(self, input_queue, input_data):
        input_queue.put(input_data)

    def run(self):
        self.asr_inference_proc.start()  # Run ASR inference
