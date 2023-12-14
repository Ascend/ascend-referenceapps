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

import json
import sys
import time

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

sys.path.append("/home/digital_human/AI_DIGITAL_HUMAN/AI_Module")

from base_model import AI_Model
from multiprocessing import Queue

_MODLE_NAME = "chatglm2-6b"
_SERVER = "71.14.88.12:8000"

class LLM_Model(AI_Model):
    def __init__(self, input_queue, output_queue):
        super().__init__(input_queue, output_queue)

    def handle(self, query, history):
        with httpclient.InferenceServerClient(_SERVER) as client:
            ts = time.time()

            input0_data = input0_data = np.array([query], dtype=np.object_)
            input1_data = np.array([json.dumps(history)], dtype=np.object_)
            inputs = [
                httpclient.InferInput(
                    "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
                ),
                httpclient.InferInput(
                    "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
                ),
            ]

            inputs[0].set_data_from_numpy(input0_data)
            inputs[1].set_data_from_numpy(input1_data)

            outputs = [
                httpclient.InferRequestedOutput("OUTPUT0"),
                httpclient.InferRequestedOutput("OUTPUT1"),
            ]

            response = client.infer(_MODLE_NAME, inputs, request_id=str(1), outputs=outputs)

            result = response.get_response()
            output0_data = response.as_numpy("OUTPUT0")
            output1_data = response.as_numpy("OUTPUT1")

            response = (output0_data.astype(np.bytes_))[0].decode("utf-8")
            history = json.loads(output1_data[0])

            latency = time.time() - ts

            return response, history
    
    def get_data(self, input_data):
        self.input_queue.put(input_data)
    
    def run(self):
        while True:
            if not self.input_queue.empty():
                input_data = self.input_queue.get()
                response, history = self.handle(input_data["query"], input_data["history"])
                self.output_queue.put({"response": response, "history": history})

    
