from pathlib import Path

import yaml

from ...utils import convert_to_chat, _make_options_str, make_random_template_data, load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples('skt/kobest_v1', split, max_examples, subset='boolq')

    def adding_columns(example):
        label = example['label']
        options = _make_options_str('거짓', '참')
        answer = ['거짓', '참'][label]
        return {
            "options": options,
            "answer": answer
        }
    new_ds = ds.map(adding_columns)

    with open(Path(__file__).parent / "template.yaml", 'rt', encoding='utf-8') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)['kobest_boolq']
    new_ds = make_random_template_data(templates, new_ds)

    new_ds = convert_to_chat(new_ds)

    return new_ds
