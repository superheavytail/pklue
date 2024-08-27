from datasets import concatenate_datasets, load_dataset

from ...utils import load_dataset_max_examples


def process(max_examples, split):
    ds = load_dataset_max_examples('heavytail/neutral_korean_multiturn_gpt4_kullm', split, max_examples)

    # concatenate ultrachat, aha, hand
    # ds = concatenate_datasets([ds['ultrachat'], ds['aha'], ds['hand']])
    #
    # if max_examples and max_examples < len(ds):
    #     ds = ds.train_test_split(train_size=max_examples)['train']
    #
    # # make it chat form
    # def worker(item):
    #     chat = []
    #     for i, utterance in enumerate(item['data']):
    #         if i % 2 == 0:
    #             turn = 'user'
    #         else:
    #             turn = 'assistant'
    #         chat.append((turn, utterance))
    #     return {'chat': chat}
    # ds = ds.map(worker, remove_columns=['data'])

    return ds
