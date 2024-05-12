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

from datasets import Dataset, load_dataset


def list_to_dataset(l, truncate=None):
    if truncate:
        l = l[:truncate]
    d = {k: [] for k in l[0].keys()}
    for i, e in enumerate(l):
        for k, v in e.items():
            d[k].append(v)
    return Dataset.from_dict(d)


def _make_options_str(*options):
    """utility function that make {options} form easily. Returns raw string."""
    l = ['선택지:']
    for option in options:
        l.append(f' - {option}')
    return '\n'.join(l)


def make_random_template_data(given_templates, data):
    def replace_xa0_if_str(x):
        if isinstance(x, str):
            x = x.replace(u'\xa0', u'')
        return x

    new_ds = Dataset.from_list([
        {k: v.format_map(
            {k2: replace_xa0_if_str(v2) for k2, v2 in item.items()}  # sometimes u'\xa0' appears, so replace it.
        ) for k, v in choice(given_templates).items()} for item in data
    ])

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


def load_dataset_max_examples(dataset_name, split=None, max_examples=None, subset: str = None):
    if subset and split:
        ds = load_dataset(dataset_name, subset, split=split)
    elif split:
        ds = load_dataset(dataset_name, split=split)
    else:
        ds = load_dataset(dataset_name)
    if max_examples and max_examples < len(ds):
        ds = ds.train_test_split(train_size=max_examples)['train']
    return ds
