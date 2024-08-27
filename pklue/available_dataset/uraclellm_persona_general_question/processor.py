from datasets import Dataset

from ...utils import load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples("heavytail/uraclellm_persona_general_question", split, max_examples)

    # change 'prompt', 'completion' column names to 'user', 'assistant' and make it chat form
    ds = Dataset.from_list([{
        'chat': [('user', e['prompt']), ('assistant', e['completion'])]
    } for e in ds])

    return ds
