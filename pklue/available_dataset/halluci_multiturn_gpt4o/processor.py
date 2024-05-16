from ...utils import load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples("heavytail/halluci_multiturn_gpt4o", split, max_examples)
    return ds