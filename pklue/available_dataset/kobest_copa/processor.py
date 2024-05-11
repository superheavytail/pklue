from pathlib import Path

import yaml

from ...utils import convert_to_chat, _make_options_str, make_random_template_data, load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples('skt/kobest_v1', split, max_examples, subset='copa')

    # add [options, euro_or_ro(으로/로), eun_or_neun(은/는), answer] columns to dataset
    def adding_columns(example):
        options = _make_options_str(
            example['alternative_1'], example['alternative_2']
        )
        question = example['question'].strip()  # since it is occasionally not stripped
        if question == '원인':
            euro_or_ro = '으로'
            eun_or_neun = '은'
        elif question == '결과':
            euro_or_ro = '로'
            eun_or_neun = '는'
        else:
            raise NotImplementedError(f"unexpected raw data question: '{example['question']}'")
        answer = example[f"alternative_{example['label'] + 1}"]
        return {
            "options": options,
            "euro_or_ro": euro_or_ro,
            "eun_or_neun": eun_or_neun,
            "answer": answer
        }
    new_ds = ds.map(adding_columns)

    with open(Path(__file__).parent / "template_copa.yaml", 'rt', encoding='utf-8') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)['kobest_copa']
    new_ds = make_random_template_data(templates, new_ds)

    new_ds = convert_to_chat(new_ds)

    return new_ds
