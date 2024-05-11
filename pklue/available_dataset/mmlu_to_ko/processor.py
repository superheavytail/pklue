from pathlib import Path

import yaml

from ...utils import make_random_template_data, convert_to_chat, load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples('heavytail/ko_mmlu', split, max_examples)

    with open(Path(__file__).parent / "template.yaml", 'rt', encoding='utf-8') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)['templates']

    # make options string
    new_ds = ds.map(
        lambda example: {'gold': example[f'{example["target"]}']}
    )

    new_ds = make_random_template_data(templates, new_ds)
    new_ds = convert_to_chat(new_ds)
    return new_ds
