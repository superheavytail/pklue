from datasets import Dataset

from ...utils import load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples("nlpai-lab/kullm3-dolly-gpt4", split, max_examples)

    # change 'instruction', 'output' column names to 'user', 'assistant' and make it dictionary form
    ds = Dataset.from_list([{
        'chat': [('user', e['instruction']), ('assistant', e['output'])]
    } for e in ds])

    return ds