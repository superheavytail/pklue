from ...utils import convert_to_chat, load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples("nlpai-lab/kullm3-personal-info", split, max_examples)

    # change 'instruction', 'output' column names to 'user', 'assistant' and make it chat form
    ds = ds.rename_columns({
        'instruction': 'prompt',
        'output': 'completion',
    })
    ds = convert_to_chat(ds)

    return ds
