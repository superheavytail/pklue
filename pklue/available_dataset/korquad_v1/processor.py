from pathlib import Path

from datasets import load_dataset
import yaml

from ...utils import convert_to_chat, make_random_template_data


def process(max_examples, split):
    ds = load_dataset("squad_kor_v1", split=split)

    if max_examples:
        ds = ds.train_test_split(train_size=max_examples)['train']

    # add 'answer' column to dataset
    new_ds = ds.map(lambda example: {'answer': example['answers']['text'][0]})

    with open(Path(__file__).parent / "template_korquad_v1.yaml", 'rt', encoding='utf-8') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)['korquad_v1']
    new_ds = make_random_template_data(templates, new_ds, max_examples)

    new_ds = convert_to_chat(new_ds)

    return new_ds
