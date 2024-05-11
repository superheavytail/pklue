from datasets import Dataset

from ...utils import load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples("vicgalle/alpaca-gpt4", split, max_examples)
    l = []
    for e in ds:
        if e['input']:
            prompt = e['instruction']
        else:
            prompt = f"{e['instruction']}\n\n{e['input']}"
        l.append({
            'chat': [
                ('user', prompt),
                ('assistant', e['output'])
            ]
        })
    ds = Dataset.from_list(l)

    return ds
