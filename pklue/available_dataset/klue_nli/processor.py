from pathlib import Path

import yaml
from datasets import Dataset

from ...utils import _make_options_str, make_random_template_data, convert_to_chat, load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples('klue', split, None, subset='nli')  # since we're gonna deduplication, get all data.

    with open(Path(__file__).parent / "template_nli.yaml", 'rt', encoding='utf-8') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)['klue_nli']

    # deduplication for nli subset. since klue-nli dataset have too many duplicated premise
    deduplicated_ds = {c: [] for c in ds.column_names}
    set_of_premise = set()
    for row in ds:
        if row['premise'] in set_of_premise:
            continue
        for k, v in row.items():
            deduplicated_ds[k].append(v)
    deduplicated_ds = Dataset.from_dict(deduplicated_ds)
    if max_examples and len(deduplicated_ds) > max_examples:
        deduplicated_ds = deduplicated_ds.train_test_split(train_size=max_examples)['train']  # apply after dedup

    # add 'options', 'answer' column to dataset
    options_str = ['수반', '중립', '모순']
    new_ds = deduplicated_ds.map(
        lambda example: {'options': _make_options_str(*options_str), 'answer': options_str[example['label']]}
    )

    new_ds = make_random_template_data(templates, new_ds)
    new_ds = convert_to_chat(new_ds)
    return new_ds
