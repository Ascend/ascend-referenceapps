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

import re


# 将文本按标点符号进行分割
def split_text(text):
    pattern = re.compile(r"[，,.。?？！!;；:：]")
    texts = pattern.split(text)

    # 将标点符号添加到文本片段末尾
    punctuations = pattern.findall(text)
    for i in range(len(texts) - 1):
        texts[i] += punctuations[i]

    # 去除末尾的空字符串
    if texts and not texts[-1]:
        texts.pop()

    return texts
