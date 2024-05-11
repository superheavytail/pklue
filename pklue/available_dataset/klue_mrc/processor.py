from pathlib import Path

import yaml
from datasets import load_dataset, Dataset

from ...utils import _make_options_str, make_random_template_data


def process(max_examples, split):
    ds = load_dataset('klue', 'mrc', split=split)

    with open(Path(__file__).parent / "template_mrc.yaml", 'rt', encoding='utf-8') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)['klue_mrc']

    # add an 'answer' column from 'answers' column
    new_ds = ds.map(
        lambda item: {'answer': item['answers']['text'][0]},
        remove_columns=['answers']  # for avoiding mistakes
    )

    new_ds = make_random_template_data(templates, new_ds, max_examples)

    return new_ds
