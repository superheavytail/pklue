from pathlib import Path

import yaml

from ...utils import make_random_template_data, convert_to_chat, load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples('klue', split, max_examples, subset='sts')

    with open(Path(__file__).parent / "template_sts.yaml", 'rt', encoding='utf-8') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)['klue_sts']

    new_ds = make_random_template_data(templates, ds)
    new_ds = convert_to_chat(new_ds)
    return new_ds
