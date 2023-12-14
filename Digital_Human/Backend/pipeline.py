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

import sys
import subprocess
from multiprocessing import Process, Queue

sys.path.append("/home/digital_human/AI_DIGITAL_HUMAN")
sys.path.append("/home/digital_human/AI_DIGITAL_HUMAN/AI_Module")

import cv2
import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import ImageSequenceClip, concatenate_videoclips, AudioFileClip

from Utils import utils
from base_model import AI_Model
from AI_Module.kws_model import KWS_Model
from AI_Module.asr_model import ASR_Model
from AI_Module.llm_model import LLM_Model
from AI_Module.tts_model import TTS_Model
from AI_Module.wav2lip_model import WAV2LIP_Model
from AI_Module.ws_receiver import WS_Receiver

""" ADH Pipeline assembles the AI modules together """

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

class ADH_Pipeline:
    def __init__(self, client_id):
        self.module_names = ["KWS", "ASR", "LLM", "TTS", "WAV2LIP"]
        self.history = []
        self.client_id = client_id
        self.kws_in_queue = Queue()
        self.kws_out_queue = Queue()
        self.export_queue = Queue()
        self.asr_in_queue = Queue()
        self.asr_out_queue = Queue()
        self.llm_in_queue = Queue()
        self.llm_out_queue = Queue()
        self.tts_in_queue = Queue()
        self.tts_out_queue = Queue()
        self.wav2lip_out_queue = Queue()
        self.count = 0
        self.ws = WS_Receiver(export_que=self.export_queue)
        self.listen_start = True

    def run_ws_receiver(self):
        # Run ws_receiver receiver module, and it will constantly receiving user's audio input from WebUI
        self.ws.run()

    def run_kws_model(self):
        # Run KWS model, and it will push the output signal to kws_out_queue
        kws_model = KWS_Model(self.kws_in_queue, self.kws_out_queue)
        kws_model.run()

    def run_asr_model(self):
        # Run ASR model, and it will push the output text to asr_out_queue
        asr_model = ASR_Model(self.asr_in_queue, self.asr_out_queue)
        asr_model.run()

    def run_llm_model(self):
        # Run LLM model, and it will push the output to llm_out_queue
        llm_model = LLM_Model(self.llm_in_queue, self.llm_out_queue)
        llm_model.run()

    def run_tts_model(self):
        # Run TTS model, and it will push the output to tts_out_queue
        tts_model = TTS_Model(self.tts_in_queue, self.tts_out_queue)
        tts_model.run()

    def run_wav2lip_model(self):
        # Run WAV2LIP model, and it will push the output to wav2lip_out_queue
        wav2lip_model = WAV2LIP_Model(self.tts_out_queue, self.wav2lip_out_queue)
        wav2lip_model.run()


    def run_pipeline(self):
        # # Run KWS model in a separate process
        # kws_process = Process(target=self.run_kws_model)
        # kws_process.start()

        # Run ws_receiver in a separate process
        ws_process = Process(target=self.run_ws_receiver)
        ws_process.start()

        # Run ASR model in a separate process
        asr_process = Process(target=self.run_asr_model)
        asr_process.start()

        # Run LLM model in a separate process
        llm_process = Process(target=self.run_llm_model)
        llm_process.start()

        # Run TTS model in a separate process
        tts_process = Process(target=self.run_tts_model)
        tts_process.start()
        
        # Run WAV2LIP model in a separate process
        wav2lip_process = Process(target=self.run_wav2lip_model)
        wav2lip_process.start()

        # Data handle from the queues
        i = 0
        j = 0
        clips = []
        cnt = 0
        while True:
            try:
                if not self.export_queue.empty():
                    waveform = self.export_queue.get()
                    asr_input = {
                        "waveform": waveform,
                        "listen_start": self.listen_start,
                    }
                    self.asr_in_queue.put(asr_input)
                    self.listen_start = False
                    
                if not self.asr_out_queue.empty():
                    asr_out_dict = self.asr_out_queue.get()
                    if "text" in asr_out_dict:
                        print("[RESULT] ASR output printing..", asr_out_dict["text"])
                        self.count += 1
                        self.llm_in_queue.put({"query": asr_out_dict["text"], "history": self.history})

                if not self.llm_out_queue.empty():
                    llm_out_data = self.llm_out_queue.get()
                    response = llm_out_data["response"]
                    self.history = llm_out_data["history"]
                    print("[LLM] Answer text from LLM Model:", response)
                    print("[LLM] Current history text length:", len(self.history))

                    texts = utils.split_text(response)
                    self.tts_in_queue.put(texts)
                    print("[TTS] Putting texts to TTS Model:", texts)

                if not self.wav2lip_out_queue.empty():
                    wav2lip_out_data = self.wav2lip_out_queue.get()
                    imgs_list, tts_wav, is_end = wav2lip_out_data
                    i += 1
                    video_clip = ImageSequenceClip(imgs_list, fps=25)
                    audio_file_str = "/home/digital_human/tts/streaming_om_output/" + str(j) + "_" + str(i-1) + ".wav"
                    audio_clip = AudioFileClip(audio_file_str)
                    video_clip = video_clip.set_audio(audio_clip)
                    video_clip = video_clip.fl_image(bgr_to_rgb)
                    clips.append(video_clip)

                    if is_end:
                        print("Concatenate videos")
                        i = 0
                        j += 1
                        final_clip = concatenate_videoclips(clips)
                        final_clip.write_videofile("./video_output/multi_round_output_" + str(cnt) + ".mp4", codec='libx264')
                        clips = []
                        cnt += 1
                        self.listen_start = True
                        print("========================== [VIDEO_SAVED] ==========================")

            except KeyboardInterrupt:
                self.ws.terminate()
                sys.exit(0)
