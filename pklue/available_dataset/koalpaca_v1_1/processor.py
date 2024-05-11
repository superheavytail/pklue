from ...utils import convert_to_chat, load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples("beomi/KoAlpaca-v1.1a", split, max_examples)
    ds = ds.select_columns(['instruction', 'output'])

    # change 'instruction', 'output' column names to 'user', 'assistant' and make it dictionary form
    ds = ds.rename_columns({
        'instruction': 'prompt',
        'output': 'completion',
    })
    ds = convert_to_chat(ds)

    return ds
