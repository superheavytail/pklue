from pathlib import Path

import yaml

from ...utils import convert_to_chat, make_random_template_data, load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples("squad_kor_v1", split, max_examples)

    # add 'answer' column to dataset
    new_ds = ds.map(lambda example: {'answer': example['answers']['text'][0]})

    with open(Path(__file__).parent / "template_korquad_v1.yaml", 'rt', encoding='utf-8') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)['korquad_v1']
    new_ds = make_random_template_data(templates, new_ds)

    new_ds = convert_to_chat(new_ds)

    return new_ds
