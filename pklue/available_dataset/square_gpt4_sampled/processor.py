from ...utils import load_dataset_max_examples, convert_to_chat


def process(max_examples, split):
    ds = load_dataset_max_examples("nlpai-lab/kullm3-square-gpt4-sampled", split, max_examples)

    # make it chat form
    ds = ds.rename_columns({
        'instruction': 'prompt',
        'output': 'completion'
    })
    ds = convert_to_chat(ds)

    return ds
