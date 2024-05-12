from pathlib import Path

import yaml

from ...utils import convert_to_chat, _make_options_str, make_random_template_data, load_dataset_max_examples
from ...korean_utils import bojosa


def process(max_examples, split):
    ds = load_dataset_max_examples('skt/kobest_v1', split, max_examples, subset='wic')

    def adding_columns(data):
        label = data['label']
        if label == 0:
            answer = '다른 뜻입니다.'
        elif label == 1:
            answer = '같은 뜻입니다.'
        else:
            raise NotImplementedError
        return {
            'answer': answer,
            'options': _make_options_str('다른 뜻입니다.', '같은 뜻입니다.'),
            'eun_or_neun': bojosa(data['word'])
        }

    new_ds = ds.map(adding_columns)

    with open(Path(__file__).parent / "template.yaml", 'rt', encoding='utf-8') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)['kobest_wic']
    new_ds = make_random_template_data(templates, new_ds)

    new_ds = convert_to_chat(new_ds)

    return new_ds
