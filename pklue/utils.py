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


def make_prompts_by_random_template_(subset, dataset_name, subset_name, template):
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


def _make_options_str(*options):
    """utility function that make {options} form easily. Returns raw string."""
    l = ['선택지:']
    for option in options:
        l.append(f' - {option}')
    return '\n'.join(l)


def make_random_template_data(given_templates, data, max_examples):
    def replace_xa0_if_str(x):
        if isinstance(x, str):
            x = x.replace(u'\xa0', u'')
        return x

    new_ds = Dataset.from_list([
        {k: v.format_map(
            {k2: replace_xa0_if_str(v2) for k2, v2 in item.items()}  # sometimes u'\xa0' appears, so replace it.
        ) for k, v in choice(given_templates).items()} for item in data
    ])

    if max_examples:
        new_ds = new_ds.train_test_split(train_size=max_examples)['train']

    return new_ds


def convert_to_chat(data: Dataset):
    original_column_names = {'prompt', 'completion'}
    assert set(data.column_names) == original_column_names
    new_data = data.map(
        lambda item: {'chat': [
            ('user', item['prompt']), ('assistant', item['completion'])
        ]},
        remove_columns=list(original_column_names)
    )
    return new_data
