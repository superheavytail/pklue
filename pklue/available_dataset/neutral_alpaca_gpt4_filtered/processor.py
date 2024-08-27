from ...utils import load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples("heavytail/neutral_alpaca_gpt4_filtered", split, max_examples)
    return ds
