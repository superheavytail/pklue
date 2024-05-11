from datasets import Dataset

from ...utils import load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples("nlpai-lab/kullm-v2", split, max_examples).remove_columns('id')

    l = []
    for e in ds:
        if not e['input']:
            prompt = e['instruction']
        else:
            prompt = f"{e['instruction']}\n\n{e['input']}"
        l.append({'chat': [('user', prompt), ('assistant', e['output'])]})
    ds = Dataset.from_list(l)

    return ds
