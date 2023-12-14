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
import time
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue

sys.path.append("/home/digital_human/AI_DIGITAL_HUMAN/Backend")

import AudioReceiver

''' WS_Receiver moduel takes audio input from WebUI and feed to the downstreams '''

class WS_Receiver():
    def __init__(self, ip_address='0.0.0.0', port=8765, segment_duration_ms=500, buffer_size=10, export_num_seg=4, export_que=None):
        self.export_que = export_que
        self.ip_address = ip_address
        self.port = port
        self.segment_duration_ms = segment_duration_ms
        self.export_num_seg = export_num_seg
        self.buffer_size = buffer_size
        self.audio_dir = '/home/digital_human/AI_DIGITAL_HUMAN/Backend/audio_files'
    
    def current_timestamp(self):
        return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def run(self):
        AudioReceiver.launch(self.ip_address, self.port, self.segment_duration_ms, self.buffer_size, self.export_num_seg, self.export_que)
    
    def terminate(self):
        AudioReceiver.shutdown()
        print('[INFO] Receiver proc has been terminated.')