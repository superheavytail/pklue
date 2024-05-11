from pathlib import Path

import yaml
from datasets import load_dataset

from ...utils import _make_options_str, make_random_template_data, convert_to_chat


def process(max_examples, split):
    ds = load_dataset('klue', 'ynat', split=split)

    with open(Path(__file__).parent / "template_ynat.yaml", 'rt', encoding='utf-8') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)['klue_ynat']

    # make options string
    options_str = ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']
    new_ds = ds.map(
        lambda example: {'options': _make_options_str(*options_str), 'answer': options_str[example['label']]}
    )

    new_ds = make_random_template_data(templates, new_ds, max_examples)
    new_ds = convert_to_chat(new_ds)
    return new_ds
