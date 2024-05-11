from datasets import load_dataset

from ...utils import convert_to_chat


def process(max_examples, split):
    ds = load_dataset("beomi/KoAlpaca-v1.1a")[split]
    ds = ds.select_columns(['instruction', 'output'])

    if max_examples:
        ds = ds.train_test_split(train_size=max_examples)['train']

    # change 'instruction', 'output' column names to 'user', 'assistant' and make it dictionary form
    ds = ds.rename_columns({
        'instruction': 'prompt',
        'output': 'completion',
    })
    ds = convert_to_chat(ds)

    return ds
