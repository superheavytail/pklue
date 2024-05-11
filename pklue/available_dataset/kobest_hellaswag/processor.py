from pathlib import Path

import yaml

from ...utils import convert_to_chat, _make_options_str, make_random_template_data, load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples('skt/kobest_v1', split, max_examples, subset='hellaswag')

    # add 'options', 'answer' column to dataset
    new_ds = ds.map(
        lambda example: {
            'options': _make_options_str(
                example['ending_1'], example['ending_2'], example['ending_3'], example['ending_4']
            ),
            'answer': example[f"ending_{example['label']+1}"]
        }
    )

    with open(Path(__file__).parent / "template_hellaswag.yaml", 'rt', encoding='utf-8') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)['kobest_hellaswag']
    new_ds = make_random_template_data(templates, new_ds)

    new_ds = convert_to_chat(new_ds)

    return new_ds
