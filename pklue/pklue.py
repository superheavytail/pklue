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
from pathlib import Path
import importlib

import datasets
from datasets import concatenate_datasets


def get_mixture(
        dataset_names: List[str],
        max_examples: int = None,
        split: str = 'train',
) -> datasets.Dataset:
    """Make mixed huggingface dataset with selected datasets.

    Args:
        dataset_names: list of dataset names. names are case-insensitive.
        max_examples: the number of maximum length of examples when do truncation.
        split: 'train' or 'test'
    Returns:
        Huggingface dataset which contains mixture of 'dataset_names'.
        Returned dataset's columns are like
        {"chat": [['user', '...'], ['assistant', '...'], ...]}
    """
    available_dataset = [e.name for e in (Path(__file__).parent / "available_dataset/").glob("*/")]
    assert all(n in available_dataset for n in dataset_names), f"Invalid dataset name. available: {available_dataset}"

    processed_datasets = []
    for dataset_name in dataset_names:
        module = importlib.import_module(f".available_dataset.{dataset_name}.processor", package='pklue')
        processed_datasets.append(module.process(max_examples, split))

    return concatenate_datasets(processed_datasets)


if __name__ == '__main__':
    raise NotImplementedError
