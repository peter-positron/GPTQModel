# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
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

from ._const import get_best_device
from .auto import MODEL_MAP, GPTQModel
from .base import BaseGPTQModel
from .definitions import *
from .gpt_bigcode import GPTBigCodeGPTQ
from .gptj import GPTJGPTQ
from .gptneox import GPTNeoXGPTQ
from .internlm import InternLMGPTQ
from .llama import LlamaGPTQ
from .longllama import LongLlamaGPTQ
from .mistral import MistralGPTQ
from .mixtral import MixtralGPTQ
from .mpt import MPTGPTQ
from .qwen import QwenGPTQ
from .qwen2 import Qwen2GPTQ
from .qwen2_moe import Qwen2MoeGPTQ
from .starcoder2 import Starcoder2GPTQ
from .dual_stream_roformer import DualStreamRoformerGPTQ

GPTQ_MODEL_LIST = [
    LlamaGPTQ,
    MistralGPTQ,
    MixtralGPTQ,
    GPTBigCodeGPTQ,
    GPTJGPTQ,
    GPTNeoXGPTQ,
    QwenGPTQ,
    Qwen2GPTQ,
    Qwen2MoeGPTQ,
    Starcoder2GPTQ,
    DualStreamRoformerGPTQ
]
