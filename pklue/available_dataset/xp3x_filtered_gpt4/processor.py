from ...utils import convert_to_chat, load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples('nlpai-lab/kullm3-xp3x-filtered-gpt4', split, max_examples)

    # make it chat form
    ds = ds.rename_columns({
        'instruction': 'prompt',
        'answer': 'completion'
    })
    ds = convert_to_chat(ds)

    return ds
