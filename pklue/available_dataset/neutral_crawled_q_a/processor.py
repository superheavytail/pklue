from itertools import chain

from datasets import Dataset, load_dataset, concatenate_datasets
from tqdm import tqdm


def process(max_examples, split):
    ds = load_dataset("heavytail/neutral_crawled_q_a")

    # change 'prompt', 'completion' column names to 'user', 'assistant' and make it chat form
    # ds = Dataset.from_list([{
    #     'chat': [('user', e['prompt']), ('assistant', e['completion'])]
    # } for e in chain(*ds_dict.values())])
    #
    # # make 20% 2-turn, 10% 3-turn.
    # ds.train_test_split()
    # split_20_percent = len(ds) // 5
    # if split_20_percent % 2 != 0:
    #     split_20_percent += 1  # so, it's able to reformat 2-turn conversation
    # split_10_percent = round(len(ds) * 0.1)
    # while split_10_percent % 3 != 0:
    #     split_10_percent += 1  # so, it's able to reformat 3-turn conversation
    # tmp = ds.train_test_split(train_size=split_20_percent)
    # for_2turn = tmp['train']
    # remainder = tmp['test']
    # tmp = remainder.train_test_split(train_size=split_10_percent)
    # for_3turn = tmp['train']
    # for_1turn = tmp['test']
    #
    # # make it to 2-turn conversation
    # print("merging chat to make 2-turn conversation...")
    # assert len(for_2turn) % 2 == 0
    # merged_2_turn = []
    # for i in tqdm(range(0, len(for_2turn), 2), desc="merging 2-turn"):
    #     first_chat = for_2turn[i]['chat']
    #     second_chat = for_2turn[i+1]['chat']
    #     merged_2_turn.append({
    #         'chat': first_chat + second_chat
    #     })
    # merged_2_turn = Dataset.from_list(merged_2_turn)
    #
    # # make it to 3-turn conversation
    # print("merging chat to make 3-turn conversation...")
    # assert len(for_3turn) % 3 == 0
    # merged_3_turn = []
    # for i in tqdm(range(0, len(for_3turn), 3), desc="merging 3-turn"):
    #     first_chat = for_3turn[i]['chat']
    #     second_chat = for_3turn[i+1]['chat']
    #     third_chat = for_3turn[i+2]['chat']
    #     merged_3_turn.append({
    #         'chat': first_chat + second_chat + third_chat
    #     })
    # merged_3_turn = Dataset.from_list(merged_3_turn)
    #
    # ds = concatenate_datasets([for_1turn, merged_2_turn, merged_3_turn])

    return ds
