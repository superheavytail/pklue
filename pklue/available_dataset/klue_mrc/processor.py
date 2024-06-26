from pathlib import Path

import yaml

from ...utils import make_random_template_data, convert_to_chat, load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples('klue', split, max_examples, subset='mrc')

    with open(Path(__file__).parent / "template_mrc.yaml", 'rt', encoding='utf-8') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)['klue_mrc']

    # add an 'answer' column from 'answers' column
    new_ds = ds.map(
        lambda item: {'answer': item['answers']['text'][0]},
        remove_columns=['answers']  # for avoiding mistakes
    )

    new_ds = make_random_template_data(templates, new_ds)
    new_ds = convert_to_chat(new_ds)
    return new_ds
