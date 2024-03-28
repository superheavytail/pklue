# Copyright 2023 NLP & AI Lab - Korea University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""processors for datasets"""
import os
import random

from datasets import load_dataset, concatenate_datasets, Dataset

from .utils import list_to_dataset
from .utils import make_prompts_by_random_template


def _kullm_v2_processor(max_examples, split):
    kullm_v2 = load_dataset('nlpai-lab/kullm-v2')[split].remove_columns('id')

    # slicing to max_examples
    if max_examples:
        kullm_v2 = kullm_v2.train_test_split(train_size=max_examples)['train']

    # if split == 'train':
    #     return kullm_v2['train']
    # elif split == 'test':
    #     return kullm_v2['test']
    return kullm_v2


def _kobest_processor(max_examples, split):
    def load_kobest_part(name):
        if split == 'train':
            d = load_dataset('skt/kobest_v1', name, split=f"train")
            if max_examples is not None and len(d) > max_examples:
                d = d.train_test_split(train_size=max_examples)['train']
            return d
        elif split == 'test':
            return load_dataset('skt/kobest_v1', name, split=f"test")

    # you can modify below
    subset_names = ['hellaswag', 'copa', 'boolq', 'sentineg', 'wic']

    # making prompts for each dataset with randomly chosen template
    # prompts_collection = {k: [] for k in subset_names}
    # for subset_name in subset_names:
    #     ds = load_kobest_part(subset_name)
    #     custom_templates = templates.datasets[subset_name]
    #     assert len(custom_templates) == 10
    #     for i, row in enumerate(ds):
    #         template = choice(custom_templates)
    #         prompt = getattr(templates, f"_process_kobest_{subset_name}")(template, **row)
    #         prompts_collection[subset_name].append(prompt)
    prompts_collection = {}
    for subset_name in subset_names:
        ds = load_kobest_part(subset_name)
        prompts = make_prompts_by_random_template(ds, 'kobest', subset_name)
        prompts_collection[subset_name] = prompts

    # concatenate all subsets into HF Dataset
    prompts_datasets = {k: list_to_dataset(v, truncate=max_examples) for k, v in prompts_collection.items()}
    concatenated = concatenate_datasets(list(prompts_datasets.values()))

    return concatenated


def _klue_processor(max_examples, split):
    def load_klue_part(name):
        if split == 'train':
            return load_dataset('klue', name, split='train')
            # if max_examples is not None and len(d) > max_examples:
            #     return d.train_test_split(train_size=max_examples)['train']
        elif split == 'test':
            return load_dataset('klue', name, split='validation')
            # return d.train_test_split(train_size=500)['train']
        else:
            raise NotImplementedError

    # you can modify below
    subset_names = ['sts', 'mrc', 'nli', 'ynat']

    prompts_collection = {}
    for subset_name in subset_names:
        ds = load_klue_part(subset_name)

        # deduplication for nli subset. since klue nli dataset have too many duplicated premise
        if subset_name == 'nli':
            deduplicated_ds = {c: [] for c in ds.column_names}
            set_of_premise = set()
            for row in ds:
                if row['premise'] in set_of_premise:
                    continue
                for k, v in row.items():
                    deduplicated_ds[k].append(v)

        prompts = make_prompts_by_random_template(ds, "klue", subset_name)
        prompts_collection[subset_name] = prompts

    # concatenate all subsets into HF Dataset
    prompts_datasets = {k: list_to_dataset(v, truncate=max_examples) for k, v in prompts_collection.items()}
    concatenated = concatenate_datasets(list(prompts_datasets.values()))

    return concatenated


def _ko_arc_processor(max_examples, split):
    hf_key = os.environ['HF_API_KEY']
    from huggingface_hub import login
    login(hf_key)
    ds = load_dataset("heavytail/ko_arc")['train']

    # slicing to max_examples
    if max_examples:
        ds = ds.train_test_split(train_size=max_examples)['train']

    prompts = make_prompts_by_random_template(ds, "ko_arc", None)
    prompts = list_to_dataset(prompts)
    return prompts


def _ko_commongenv2_processor(max_examples, split):
    hf_key = os.environ['HF_API_KEY']
    from huggingface_hub import login
    login(hf_key)
    ds = load_dataset("heavytail/ko_commongenv2")['train']

    # slicing to max_examples
    if max_examples:
        ds = ds.train_test_split(train_size=max_examples)['train']

    prompts = make_prompts_by_random_template(ds, "ko_commongenv2", None)
    prompts = list_to_dataset(prompts)
    return prompts


def _ko_mmlu_processor(max_examples, split):
    hf_key = os.environ['HF_API_KEY']
    from huggingface_hub import login
    login(hf_key)
    ds = load_dataset("heavytail/ko_mmlu")['train']

    # slicing to max_examples
    if max_examples:
        ds = ds.train_test_split(train_size=max_examples)['train']

    prompts = make_prompts_by_random_template(ds, "ko_mmlu", None)
    prompts = list_to_dataset(prompts)
    return prompts


def _ko_truthfulqa_processor(max_examples, split):
    hf_key = os.environ['HF_API_KEY']
    from huggingface_hub import login
    login(hf_key)
    ds = load_dataset("heavytail/ko_truthfulqa")['train']

    # slicing to max_examples
    if max_examples:
        ds = ds.train_test_split(train_size=max_examples)['train']

    prompts = make_prompts_by_random_template(ds, "ko_truthfulqa", None)
    prompts = list_to_dataset(prompts)
    return prompts


def _korquad_v1_processor(max_examples, split):
    ds = load_dataset("squad_kor_v1")[split]

    # slicing to max_examples
    if max_examples:
        ds = ds.train_test_split(train_size=max_examples)['train']

    prompts = make_prompts_by_random_template(ds, "korquad_v1", None)
    prompts = list_to_dataset(prompts)
    return prompts


# === for developing KULLM3 ===
def _kullm3_rename(ds, max_examples):
    """kullm3 데이터셋에서 공통된 작업

    column name을 'answer'에서 'output'으로 바꾸고 max_example 적용"""
    if max_examples:
        ds = ds.train_test_split(train_size=max_examples)['train']
    if 'answer' in ds.column_names:
        ds = ds.rename_column("answer", "output")
    return ds


def _kullm3_alpaca_gpt4_processor(max_examples, split):
    # ds = load_dataset("nlpai-lab/kullm3-alpaca-gpt4")[split]
    # l = []
    # for e in ds:
    #     r = random.random()
    #     # 임의 확률에 따라 무작위로 instruction과 input 순서 및 공백 변경
    #     if r < 0.4:
    #         inst = f"{e['instruction']}\n{e['input']}"
    #     elif r < 0.7:
    #         inst = f"{e['instruction']} {e['input']}"
    #     elif r < 0.96:
    #         inst = f"{e['instruction']} \n\n{e['input']}\n"
    #     else:
    #         inst = f"{e['input']}\n{e['instruction']}"
    #     l.append({
    #         'instruction': inst,
    #         'answer': e['answer']
    #     })
    # ds = Dataset.from_list(l)
    ds = load_dataset("nlpai-lab/kullm3-alpaca-gpt4")[split]
    return _kullm3_rename(ds, max_examples)


def _kullm3_xp3x_filtered_gpt4_processor(max_examples, split):
    ds = load_dataset("nlpai-lab/kullm3-xp3x-filtered-gpt4")[split]
    return _kullm3_rename(ds, max_examples)


def _kullm3_dolly_gpt4_processor(max_examples, split):
    ds = load_dataset("nlpai-lab/kullm3-dolly-gpt4")[split]
    return _kullm3_rename(ds, max_examples)


def _kullm3_aya_processor(max_examples, split):
    ds = load_dataset("nlpai-lab/kullm3-aya")[split]
    return _kullm3_rename(ds, max_examples)


def _koalpaca_v1_1_processor(max_examples, split):
    ds = load_dataset("beomi/KoAlpaca-v1.1a")[split]
    ds = ds.select_columns(['instruction', 'output'])
    return _kullm3_rename(ds, max_examples)


def _alpaca_gpt4_processor(max_examples, split):
    ds = load_dataset("vicgalle/alpaca-gpt4")[split]
    l = []
    for e in ds:
        if e['input']:
            instruction = e['instruction']
        else:
            instruction = f"{e['instruction']}\n\n{e['input']}"
        l.append({
            'instruction': instruction,
            'output': e['output']
        })
    return _kullm3_rename(Dataset.from_list(l), max_examples)


def _kullm3_personal_info_processor(max_examples, split):
    ds = load_dataset("nlpai-lab/kullm3-personal-info")[split]
    return _kullm3_rename(ds, max_examples)


def _kullm3_square_gpt4_sampled_processor(max_examples, split):
    ds = load_dataset("nlpai-lab/kullm3-square-gpt4-sampled")[split]
    return _kullm3_rename(ds, max_examples)
