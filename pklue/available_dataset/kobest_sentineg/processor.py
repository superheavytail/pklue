from pathlib import Path

import yaml

from ...utils import convert_to_chat, _make_options_str, make_random_template_data, load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples('skt/kobest_v1', split, max_examples, subset='sentineg')

    def adding_columns(data):
        label = data['label']
        return {
            'options': _make_options_str('부정', '긍정'),
            'answer': ['부정', '긍정'][label]
        }
    new_ds = ds.map(adding_columns)

    with open(Path(__file__).parent / "template.yaml", 'rt', encoding='utf-8') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)['kobest_sentineg']
    new_ds = make_random_template_data(templates, new_ds)

    new_ds = convert_to_chat(new_ds)

    return new_ds
