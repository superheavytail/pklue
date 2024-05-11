import random
from pathlib import Path

import yaml
from datasets import load_dataset, Dataset


def load_klue_part(name, split):
    if split == 'train':
        return load_dataset('klue', name, split='train')
        # if max_examples is not None and len(d) > max_examples:
        #     return d.train_test_split(train_size=max_examples)['train']
    elif split == 'test':
        return load_dataset('klue', name, split='validation')
        # return d.train_test_split(train_size=500)['train']
    else:
        raise NotImplementedError


def make_klue_data_by_random_template(ds, subset_name):
    with open(Path(__file__).parent / f"template_{subset_name}.yaml", 'rt') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)

    if subset_name == 'sts':
        # select only the item that 'binary_label' == 1
        ds = ds.filter(lambda item: item['binary-label'] == 1)


        for item in ds:
            selected_template = random.choice(templates['klue_sts'])
            mapped = {k: v.format_map(item) for k, v in selected_template.items()}


# Refactoring... this would be deleted in future version.
def processor(max_examples, split):
    # you can modify below
    subset_names = ['sts', 'mrc', 'nli', 'ynat']

    prompts_collection = {}
    for subset_name in subset_names:
        ds = load_klue_part(subset_name, split)

        # deduplication for nli subset. since klue nli dataset have too many duplicated premise
        if subset_name == 'nli':
            deduplicated_ds = {c: [] for c in ds.column_names}
            set_of_premise = set()
            for row in ds:
                if row['premise'] in set_of_premise:
                    continue
                for k, v in row.items():
                    deduplicated_ds[k].append(v)
            ds = Dataset.from_dict(deduplicated_ds)

        # make data with random prompt
        prompts = make_klue_data_by_random_template(ds, subset_name)
        prompts_collection[subset_name] = prompts

    # concatenate all subsets into HF Dataset
    prompts_datasets = {k: list_to_dataset(v, truncate=max_examples) for k, v in prompts_collection.items()}
    concatenated = concatenate_datasets(list(prompts_datasets.values()))

    return concatenated