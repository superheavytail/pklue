from pathlib import Path
import random

from datasets import Dataset
import yaml


from ...utils import load_dataset_max_examples


def _process_ko_arc(template, **raw_data):

    return {k: v.format_map(raw_data) for k, v in template.items()}


def process(max_examples, split):
    ds = load_dataset_max_examples("heavytail/ko_arc", split, max_examples)

    # apply random template
    with open(Path(__file__).parent / "template.yaml", 'rt') as f:
        templates = yaml.load(f, Loader=yaml.BaseLoader)

    # Explain:
    # raw_data:
    #  {'query': 'George는 손을 금방 따뜻하게 하기 위해 문지르는 중입니다. 어떤 피부 표면이 가장 많은 열을 발생시킬까요?',
    #  'response': '건조한 손바닥'}
    new_ds = [{k: v.format_map(raw_data) for k, v in random.choice(templates['template']).items()} for raw_data in ds]
    new_ds = [{'chat': [('user', e['instruction']), ('assistant', e['output'])]} for e in new_ds]
    new_ds = Dataset.from_list(new_ds)

    return new_ds
