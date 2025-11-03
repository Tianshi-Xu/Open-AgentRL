# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

# setup.py is the fallback installation script when pyproject.toml does not work
import os
from pathlib import Path

from setuptools import find_packages, setup

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "verl/version/version")) as f:
    __version__ = f.read().strip()

install_requires = [
    "accelerate>=1.11.0",
    "anthropic>=0.72.0",
    "codetiming>=1.4.0",
    "datasets>=4.3.0",
    "dill>=0.4.0",
    "fastapi[standard]>=0.115.0",
    "grpcio>=1.62.1",
    "hf-transfer>=0.1.9",
    "hydra-core>=1.3.2",
    "liger-kernel>=0.6.3",
    "math-verify>=0.8.0",
    "mathruler>=0.1.0",
    "numpy>=2.2.6",
    "nvidia-ml-py>=12.560.30",
    "opencv-fixer>=0.2.5",
    "opencv-python>=4.12.0.88",
    "optree>=0.13.0",
    "pandas",
    "peft>=0.17.1",
    "pyarrow>=22.0.0",
    "pybind11>=3.0.1",
    "pydantic>=2.9",
    "pylatexenc>=2.10",
    "qwen-vl-utils>=0.0.14",
    "ray[default]>=2.47.1",
    "tensorboard>=2.20.0",
    "tensordict>=0.6.2",
    "torch==2.6.0",
    "torchaudio>=2.6.0",
    "torchdata>=0.11.0",
    "torchvision>=0.21.0",
    "transformers>=4.57.1",
    "vllm==0.8.5.post1",
    "wandb>=0.22.3",
    "weave>=0.52.14",
    "omegaconf",
    "antlr4-python3-runtime==4.9.3",
    "flashinfer-python==0.2.9rc2",
    "setuptools",
    "docstring_parser",
    "latex2sympy2_extended",
    "pytz",
    "python-dateutil",
    "multiprocess",
]

TEST_REQUIRES = ["pytest", "pre-commit", "py-spy", "pytest-asyncio"]
PRIME_REQUIRES = ["pyext"]
GEO_REQUIRES = ["mathruler", "torchvision", "qwen_vl_utils"]
GPU_REQUIRES = ["liger-kernel", "flash-attn"]
MATH_REQUIRES = ["math-verify"]  # Add math-verify as an optional dependency
VLLM_REQUIRES = ["tensordict>=0.8.0,<=0.9.1,!=0.9.0", "vllm>=0.7.3,<=0.9.1"]
SGLANG_REQUIRES = [
    "tensordict>=0.8.0,<=0.9.1,!=0.9.0",
    "sglang[srt,openai]==0.4.10.post2",
    "torch==2.6.0",
]
TRL_REQUIRES = ["trl<=0.9.6"]
MCORE_REQUIRES = ["mbridge"]

extras_require = {
    "test": TEST_REQUIRES,
    # "prime": PRIME_REQUIRES,
    "geo": GEO_REQUIRES,
    "gpu": GPU_REQUIRES,
    "math": MATH_REQUIRES,
    "vllm": VLLM_REQUIRES,
    # "sglang": SGLANG_REQUIRES,
    "trl": TRL_REQUIRES,
    "mcore": MCORE_REQUIRES,
}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="verl",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    url="https://github.com/volcengine/verl",
    license="Apache 2.0",
    author="Bytedance - Seed - MLSys",
    author_email="zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk",
    description="verl: Volcano Engine Reinforcement Learning for LLM",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
