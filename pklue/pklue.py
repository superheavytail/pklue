# Copyright 2023 NLP & AI Lab - Korea University
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

from typing import List

import datasets
from datasets import concatenate_datasets

from . import processors

AVAILABLE_DATASETS = ['kullm_v2', 'kobest', 'klue', 'ko_arc', 'ko_commongenv2', 'ko_mmlu', 'ko_truthfulqa', 'korquad_v1',
                      'kullm3_alpaca_gpt4', 'kullm3_xp3x_filtered_gpt4', 'kullm3_dolly_gpt4', 'kullm3_aya',
                      'kullm3_personal_info', 'kullm3_square_gpt4_sampled',
                      'koalpaca_v1_1', 'alpaca_gpt4']


def get_mixture(
        dataset_names: List[str],
        max_examples: int = None,
        split: str = 'train',
        verbose: bool = False) -> datasets.Dataset:
    """Make mixed huggingface dataset with selected datasets.

    Args:
        dataset_names: list of dataset names. names are case-insensitive.
        max_examples: the number of maximum length of examples when do truncation.
        split: 'train' or 'test'
        verbose: if set True, it prints debugging message.
    Returns:
        Huggingface dataset which contains mixture of 'dataset_names'.
        Returned dataset's columns are like {"instruction", "input", "output"}
    """
    assert all(n.lower() in AVAILABLE_DATASETS for n in dataset_names), "Invalid dataset name"

    processed_datasets = [getattr(processors, f"_{d}_processor")(max_examples, split) for d in dataset_names]

    # return concatenate_datasets(processed_datasets).shuffle()
    return concatenate_datasets(processed_datasets)


if __name__ == '__main__':
    raise NotImplementedError
