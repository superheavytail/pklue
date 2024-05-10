from datasets import load_dataset, Dataset


def process(max_examples, split):
    ds = load_dataset("vicgalle/alpaca-gpt4")[split]
    l = []
    for e in ds:
        if e['input']:
            prompt = e['instruction']
        else:
            prompt = f"{e['instruction']}\n\n{e['input']}"
        l.append({
            'prompt': prompt,
            'output': e['output']
        })
    ds = Dataset.from_list(l)

    if max_examples:
        ds = ds.train_test_split(train_size=max_examples)['train']

    return ds