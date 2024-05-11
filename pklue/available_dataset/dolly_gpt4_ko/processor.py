from datasets import load_dataset, Dataset


def process(max_examples, split):
    ds = load_dataset("nlpai-lab/kullm3-dolly-gpt4")[split]

    if max_examples:
        ds = ds.train_test_split(train_size=max_examples)['train']

    # change 'instruction', 'output' column names to 'user', 'assistant' and make it dictionary form
    ds = Dataset.from_list([{
        'chat': [('user', e['instruction']), ('assistant', e['output'])]
    } for e in ds])

    return ds