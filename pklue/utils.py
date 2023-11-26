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

"""utility functions"""

from random import choice

from datasets import Dataset

from . import templates


def list_to_dataset(l, truncate=None):
    if truncate:
        l = l[:truncate]
    d = {k: [] for k in l[0].keys()}
    for i, e in enumerate(l):
        for k, v in e.items():
            d[k].append(v)
    return Dataset.from_dict(d)


def make_prompts_by_random_template(subset, dataset_name, subset_name):
    # making prompts for each dataset with randomly chosen template
    prompts = []
    if subset_name:
        custom_templates = templates.datasets[f"{dataset_name}_{subset_name}"]
    else:
        custom_templates = templates.datasets[f"{dataset_name}"]

    for i, row in enumerate(subset):
        template = choice(custom_templates)

        if subset_name:
            prompt = getattr(templates, f"_process_{dataset_name}_{subset_name}")(template, **row)
        else:
            prompt = getattr(templates, f"_process_{dataset_name}")(template, **row)

        if prompt is not None:
            prompts.append(prompt)
    return prompts
