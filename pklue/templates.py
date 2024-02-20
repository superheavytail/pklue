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

"""데이터셋에 따른 prompt 집합.

반드시 이 파일에 있는 prompt를 불러와서 사용해야 함. 하드코딩은 허용되지 않음.

What is identical to FLAN:
    - instruction에 옵션이 주어짐
    - 'input' key는 사용하지 않음
    - 모든 instruction variation에 대해 output은 같음.
    - 지금은 데이터셋당 5개의 instruction밖에 가지고 있지 않으나, 10개로 늘려야 함.

What is different with FLAN:
    - FLAN은 모든 template에 대해 {options_}, {answer} key를 사용해서 option과 answer를 주도록 통일했으나,
        여기서는 데이터셋에 주어진 key를 활용해서 명시적으로 template을 만듦

{options} format (strictly restricted):
    선택지:
     - options1
     - options2
     ...
"""

import copy

from .korean_utils import bojosa


datasets = {
    # required keys::
    # context, options, label, ending_1, ending_2, ending_3, ending_4, answer
    "kobest_hellaswag": [
        {
            'instruction': '다음의 글을 읽고 물음에 답하세요.\n\n{context}\n\n글에서 이어질 문장으로 가장 올바른 것은?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{context}"\n이어질 말로 제일 적당한 문장을 골라 줘.\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{context}"\n에 가장 어울리는 다음 말은?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{options}\n주어진 선택지 중 다음 단락에 이어지기에 가장 자연스러운 것은?\n{context}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}\n\n주어진 지문을 계속 쓴다면 다음에 올 말을 선택해.\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '아래 단락 이후에 어떤 일이 일어날까?\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '글에 이어서 문장을 더 써줘.\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '가장 합리적인 다음 문장은?\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '이 이야기가 어떻게 끝날까??\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{answer}\n이 문장 이전에 있을 법한 이야기를 써 줘.',
            'input': '',
            'output': '{context}'
        },
    ],
    # required keys::
    # context, options, label, ending_1, ending_2, ending_3, ending_4, answer
    "hyundai_human_kobest_hellaswag": [
        {
            'instruction': '아래는 한 문장의 일부분입니다. 주어진 선택지 중에서, 가장 잘 이어지는 문장을 찾아주세요.\n\n{context}\n\n가장 자연스럽게 이어지는 선택지는 무엇일까요?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 문단을 주의 깊게 읽고, 선택지 중에서 가장 논리적으로 이어질 수 있는 문장을 찾아주세요.\n\n{context}\n\n아래에서 가장 적절한 후속 문장을 고르세요.\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 문맥 {context}에 가장 적절한 대답을 선택해보세요. {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제시된 상황 {context}에서 이어질 수 있는 가장 자연스러운 대화를 선택해주세요. {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}\n\n텍스트를 보고, 이 후 어떤 사건이 일어날 것인지 예상하여 가장 잘 이어질만한 남은 부분을 선택하십시오.\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}\n\n제공된 내용을 바탕으로 어떤 사건이 계속될 것인지 추론하고 그에 따른 적절한 선택지를 고르세요.\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 맥락과 옵션을 참조하여 가장 합리적인 후속 텍스트를 선택해 주세요.\n{context}\n{options}", "output": "{answer}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '문맥과 가능한 선택사항을 분석하여 가장 적절한 텍스트를 골라주세요.\n{context}\n{options}", "output": "{answer}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제공된 문맥과 옵션을 활용하여 가능한 이야기의 끝맺음을 예측하세요.\n{context}\n{options}", "output": "{answer}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '이 주어진 배경과 선택사항들을 토대로 가장 그럴듯한 시나리오의 결말을 유추해 보세요.\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # context, options, label, ending_1, ending_2, ending_3, ending_4, answer
    "hyundai_nonhuman_kobest_hellaswag": [
        {
            'instruction': '주어진 {context}에 따라 가장 적절하게 매칭되는 옵션을 {options}에서 찾아주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context} 상황에 가장 적절한 선택을 {options}에서 결정해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}를 분석하고, 주어진 {options} 중 가장 적절한 것을 판단해보세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}를 주의 깊게 살펴보고, 제공된 {options} 중에서 최선의 선택을 결정하십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '위 주어진 {context}를 기반으로 {options} 중에서 가장 적절한 답변을 고르세요',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {context}에 따라 {options} 중에서 가장 맞는 케이스를 선택하세요',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {context}를 바탕으로 가능한 {options} 중에서 가장 올바른 것을 선택해 쓰세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}의 내용을 참조하여 가능한 {options} 중 가장 적합한 것을 선택하여 알려주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}를 참고하여 상황을 이해하고, 주어진 {options} 중에서 가장 적합한 선택을 판단하고 이를 요약형태로 표현해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}를 이용하여 현재 상황을 숙지하고, 주어진 {options} 중에서 가장 적절한 하나를 선택하여 요약문으로 제시해주세요.',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # context, options, label, ending_1, ending_2, ending_3, ending_4, answer
    "hyundai_human_agg_kobest_hellaswag": [
        {
            'instruction': '아래는 한 문장의 일부분입니다. 주어진 선택지 중에서, 가장 잘 이어지는 문장을 찾아주세요.\n\n{context}\n\n가장 자연스럽게 이어지는 선택지는 무엇일까요?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 문단을 주의 깊게 읽고, 선택지 중에서 가장 논리적으로 이어질 수 있는 문장을 찾아주세요.\n\n{context}\n\n아래에서 가장 적절한 후속 문장을 고르세요.\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 문맥 {context}에 가장 적절한 대답을 선택해보세요. {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제시된 상황 {context}에서 이어질 수 있는 가장 자연스러운 대화를 선택해주세요. {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}\n\n텍스트를 보고, 이 후 어떤 사건이 일어날 것인지 예상하여 가장 잘 이어질만한 남은 부분을 선택하십시오.\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}\n\n제공된 내용을 바탕으로 어떤 사건이 계속될 것인지 추론하고 그에 따른 적절한 선택지를 고르세요.\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 맥락과 옵션을 참조하여 가장 합리적인 후속 텍스트를 선택해 주세요.\n{context}\n{options}", "output": "{answer}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '문맥과 가능한 선택사항을 분석하여 가장 적절한 텍스트를 골라주세요.\n{context}\n{options}", "output": "{answer}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제공된 문맥과 옵션을 활용하여 가능한 이야기의 끝맺음을 예측하세요.\n{context}\n{options}", "output": "{answer}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '이 주어진 배경과 선택사항들을 토대로 가장 그럴듯한 시나리오의 결말을 유추해 보세요.\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음의 글을 읽고 물음에 답하세요.\n\n{context}\n\n글에서 이어질 문장으로 가장 올바른 것은?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{context}"\n이어질 말로 제일 적당한 문장을 골라 줘.\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{context}"\n에 가장 어울리는 다음 말은?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{options}\n주어진 선택지 중 다음 단락에 이어지기에 가장 자연스러운 것은?\n{context}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}\n\n주어진 지문을 계속 쓴다면 다음에 올 말을 선택해.\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '아래 단락 이후에 어떤 일이 일어날까?\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '글에 이어서 문장을 더 써줘.\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '가장 합리적인 다음 문장은?\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '이 이야기가 어떻게 끝날까??\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{answer}\n이 문장 이전에 있을 법한 이야기를 써 줘.',
            'input': '',
            'output': '{context}'
        },
    ],
    # required keys::
    # context, options, label, ending_1, ending_2, ending_3, ending_4, answer
    "hyundai_nonhuman_agg_kobest_hellaswag": [
        {
            'instruction': '주어진 {context}에 따라 가장 적절하게 매칭되는 옵션을 {options}에서 찾아주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context} 상황에 가장 적절한 선택을 {options}에서 결정해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}를 분석하고, 주어진 {options} 중 가장 적절한 것을 판단해보세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}를 주의 깊게 살펴보고, 제공된 {options} 중에서 최선의 선택을 결정하십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '위 주어진 {context}를 기반으로 {options} 중에서 가장 적절한 답변을 고르세요',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {context}에 따라 {options} 중에서 가장 맞는 케이스를 선택하세요',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {context}를 바탕으로 가능한 {options} 중에서 가장 올바른 것을 선택해 쓰세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}의 내용을 참조하여 가능한 {options} 중 가장 적합한 것을 선택하여 알려주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}를 참고하여 상황을 이해하고, 주어진 {options} 중에서 가장 적합한 선택을 판단하고 이를 요약형태로 표현해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}를 이용하여 현재 상황을 숙지하고, 주어진 {options} 중에서 가장 적절한 하나를 선택하여 요약문으로 제시해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음의 글을 읽고 물음에 답하세요.\n\n{context}\n\n글에서 이어질 문장으로 가장 올바른 것은?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{context}"\n이어질 말로 제일 적당한 문장을 골라 줘.\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{context}"\n에 가장 어울리는 다음 말은?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{options}\n주어진 선택지 중 다음 단락에 이어지기에 가장 자연스러운 것은?\n{context}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}\n\n주어진 지문을 계속 쓴다면 다음에 올 말을 선택해.\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '아래 단락 이후에 어떤 일이 일어날까?\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '글에 이어서 문장을 더 써줘.\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '가장 합리적인 다음 문장은?\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '이 이야기가 어떻게 끝날까??\n{context}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{answer}\n이 문장 이전에 있을 법한 이야기를 써 줘.',
            'input': '',
            'output': '{context}'
        },
    ],
    # required keys::
    # context, options, label, ending_1, ending_2, ending_3, ending_4, answer
    "hyundai_oneprompt_kobest_hellaswag": [
        {
            'instruction': 'context: {context}\n\n{options}\n\nAnswer:',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # premise, question, options, euro_or_ro(으로/로), eun_or_neun(은/는), answer
    "kobest_copa": [
        {
            'instruction': '다음 상황이 주어졌을 때, 이 상황의 {question}{euro_or_ro} 적절한 것을 고르시오.\n상황: {premise}\n\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}\n위 사건의 {question}{eun_or_neun}?\n상황: {premise}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{premise}"가 일어나게 된 {question}{eun_or_neun}??\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 현상의 {question}{euro_or_ro} 더 적절한 것을 골라줘.\n{premise}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{options}\n\n둘 중에 {premise}의 {question}인 것은 무엇인가?',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}의 {question} 생성해\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '현상: {premise}\n{question}:\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '아래 글을 읽고 물음에 답하시오.\n{premise}\n{question}{eun_or_neun} 무엇인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '가장 합리적인 선택지를 골라.\n"{premise}"의 {question}{eun_or_neun}?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{answer}"가 {question}인 사건을 생성해 줘.',
            'input': '',
            'output': '{premise}'
        },
    ],
    # required keys::
    # premise, question, options, euro_or_ro(으로/로), eun_or_neun(은/는), answer
    "hyundai_human_kobest_copa": [
        {
            'instruction': '아래의 상황을 신중하게 고려하여, {question}에 대한 가장 적절한 응답을 선택해주세요. \n\n고려해야할 상황: {premise}\n\n선택해야 할 옵션: {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '비평적 사고를 활용하여 제시된 상황 {premise}을 분석하고, 그에 의거해 {question}에 대해 가장 적합한 답변을 선택해주세요.\n\n선택 가능한 답변들:  {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '아래에 제공된 상황 내용을 바탕으로 {question}에 가장 적절한 대답을 선택하십시오.\n{premise}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 상황 설명을 기반으로 {question}에 대한 최상의 응답을 선택해 주세요.\n{premise}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {options} 내에서, {premise}를 고려하여 {question}에 가장 잘 맞는 답변을 선택해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {options} 중에서, {premise}에 기반하여 {question}에 가장 타당한 답을 선택하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 현황: {premise} 다음의 질문에 대답해보세요: {question} 가능한 선택지: {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제시된 상황은 다음과 같습니다: {premise}. 이에 대해 궁금한 것은: {question}. 선택해야 할 대답은 다음과 같습니다: {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}라는 상황이 주어졌을 때, {question}에 대한 최적의 답변을 결정하고, 그 답변을 {options} 중에서 선택하십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 문장인 {premise}를 바탕으로, {question}의 해답을 결정하고, 가능한 답변인 {options} 중에서 어떤 것이 최선인지 선택하십시오.',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # premise, question, options, euro_or_ro(으로/로), eun_or_neun(은/는), answer
    "hyundai_nonhuman_kobest_copa": [
        {
            'instruction': '주어진 사전사실 {premise}를 바탕으로 질문 {question}에 대한 답을 고르되, 선별한 답이 {options} 중 하나이어야 합니다.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}로 제공된 사실에 기초하여 {question}를 해결하되, 답안은 주어진 {options} 중 하나를 골라주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}의 내용을 철저히 이해하신 뒤, {question}에 대한 가장 타당한 답안을 {options} 중에서 결정하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}를 통해 얻은 정보를 바탕으로, {question}에 가장 적합한 해답을 {options} 중에 선택하십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}에 근거하여 기본적인 원인을 분석하고, 제공된 {options} 중에서 최적의 설명을 찾아내십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}를 바탕으로 중심적인 원인을 검토하고, 제공된 {options} 중에서 가장 타당한 설명을 고르십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {premise}를 바탕으로 {question}에 어울리는 답을 {options} 중에서 선택하여 제시해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {premise}를 참고하여, {question}에 가장 잘 맞는 대답을 {options} 중에서 골라서 제공해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': ' 주어진 {premise}를 바탕으로, {question}에 대한 가장 직접적인 응답을 찾으세요. {options} 중에서 최적의 답을 선택하십시오',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}를 참조하여, {question}의 가장 정확한 대답을 확인하세요. {options} 중에서 가장 적합한 답을 결정하십시오',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # premise, question, options, euro_or_ro(으로/로), eun_or_neun(은/는), answer
    "hyundai_human_agg_kobest_copa": [
        {
            'instruction': '아래의 상황을 신중하게 고려하여, {question}에 대한 가장 적절한 응답을 선택해주세요. \n\n고려해야할 상황: {premise}\n\n선택해야 할 옵션: {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '비평적 사고를 활용하여 제시된 상황 {premise}을 분석하고, 그에 의거해 {question}에 대해 가장 적합한 답변을 선택해주세요.\n\n선택 가능한 답변들:  {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '아래에 제공된 상황 내용을 바탕으로 {question}에 가장 적절한 대답을 선택하십시오.\n{premise}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 상황 설명을 기반으로 {question}에 대한 최상의 응답을 선택해 주세요.\n{premise}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {options} 내에서, {premise}를 고려하여 {question}에 가장 잘 맞는 답변을 선택해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {options} 중에서, {premise}에 기반하여 {question}에 가장 타당한 답을 선택하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 현황: {premise} 다음의 질문에 대답해보세요: {question} 가능한 선택지: {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제시된 상황은 다음과 같습니다: {premise}. 이에 대해 궁금한 것은: {question}. 선택해야 할 대답은 다음과 같습니다: {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}라는 상황이 주어졌을 때, {question}에 대한 최적의 답변을 결정하고, 그 답변을 {options} 중에서 선택하십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 문장인 {premise}를 바탕으로, {question}의 해답을 결정하고, 가능한 답변인 {options} 중에서 어떤 것이 최선인지 선택하십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 상황이 주어졌을 때, 이 상황의 {question}{euro_or_ro} 적절한 것을 고르시오.\n상황: {premise}\n\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}\n위 사건의 {question}{eun_or_neun}?\n상황: {premise}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{premise}"가 일어나게 된 {question}{eun_or_neun}??\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 현상의 {question}{euro_or_ro} 더 적절한 것을 골라줘.\n{premise}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{options}\n\n둘 중에 {premise}의 {question}인 것은 무엇인가?',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}의 {question} 생성해\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '현상: {premise}\n{question}:\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '아래 글을 읽고 물음에 답하시오.\n{premise}\n{question}{eun_or_neun} 무엇인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '가장 합리적인 선택지를 골라.\n"{premise}"의 {question}{eun_or_neun}?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{answer}"가 {question}인 사건을 생성해 줘.',
            'input': '',
            'output': '{premise}'
        },
    ],
    # required keys::
    # premise, question, options, euro_or_ro(으로/로), eun_or_neun(은/는), answer
    "hyundai_nonhuman_agg_kobest_copa": [
        {
            'instruction': '주어진 사전사실 {premise}를 바탕으로 질문 {question}에 대한 답을 고르되, 선별한 답이 {options} 중 하나이어야 합니다.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}로 제공된 사실에 기초하여 {question}를 해결하되, 답안은 주어진 {options} 중 하나를 골라주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}의 내용을 철저히 이해하신 뒤, {question}에 대한 가장 타당한 답안을 {options} 중에서 결정하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}를 통해 얻은 정보를 바탕으로, {question}에 가장 적합한 해답을 {options} 중에 선택하십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}에 근거하여 기본적인 원인을 분석하고, 제공된 {options} 중에서 최적의 설명을 찾아내십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}를 바탕으로 중심적인 원인을 검토하고, 제공된 {options} 중에서 가장 타당한 설명을 고르십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {premise}를 바탕으로 {question}에 어울리는 답을 {options} 중에서 선택하여 제시해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {premise}를 참고하여, {question}에 가장 잘 맞는 대답을 {options} 중에서 골라서 제공해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': ' 주어진 {premise}를 바탕으로, {question}에 대한 가장 직접적인 응답을 찾으세요. {options} 중에서 최적의 답을 선택하십시오',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}를 참조하여, {question}의 가장 정확한 대답을 확인하세요. {options} 중에서 가장 적합한 답을 결정하십시오',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 상황이 주어졌을 때, 이 상황의 {question}{euro_or_ro} 적절한 것을 고르시오.\n상황: {premise}\n\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}\n위 사건의 {question}{eun_or_neun}?\n상황: {premise}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{premise}"가 일어나게 된 {question}{eun_or_neun}??\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 현상의 {question}{euro_or_ro} 더 적절한 것을 골라줘.\n{premise}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{options}\n\n둘 중에 {premise}의 {question}인 것은 무엇인가?',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}의 {question} 생성해\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '현상: {premise}\n{question}:\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '아래 글을 읽고 물음에 답하시오.\n{premise}\n{question}{eun_or_neun} 무엇인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '가장 합리적인 선택지를 골라.\n"{premise}"의 {question}{eun_or_neun}?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{answer}"가 {question}인 사건을 생성해 줘.',
            'input': '',
            'output': '{premise}'
        },
    ],
    # required keys::
    # premise, question, options, euro_or_ro(으로/로), eun_or_neun(은/는), answer
    "hyundai_oneprompt_kobest_copa": [
        {
            'instruction': '{premise}\n\n{question}\n\n{options}',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # paragraph, question, options, answer
    # answer = {참, 거짓}
    "kobest_boolq": [
        {
            'instruction': '{paragraph}\n윗글로 미루어볼 때 다음 문장은 참인가 거짓인가?\n{question}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}\n가 주어졌을 때\n{question}\n을 판단해주세요. \n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{paragraph}"는 "{question}"을 함의한다. 진위 여부를 판별하면? \n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{options}\n{paragraph}\n안에는\n{question}\n라는 내용이 들어가 있다. 참 또는 거짓으로 대답해.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '<보기>\n{paragraph}\n다음 문장은 참인가 거짓인가?\n{question}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '지문:\n{paragraph}\n질문:\n{question}\n{options}\n정답:',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}\n"{question}"는 참인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '질문에 답하되, 지문에 근거하여 판단하세요.\n{question}은 옳은가?\n{paragraph}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 글\n{paragraph}\n을 보고 생각했을 때,\n{question}\n은 참이니?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '참, 거짓 여부 판별\n\n근거: {paragraph}\n주장 또는 질문: {question}\n{options}',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # paragraph, question, options, answer
    # answer = {참, 거짓}
    "hyundai_human_kobest_boolq": [
        {
            'instruction': '{paragraph}를 참조하여 {question}가 정확한지 여부를 판단하세요. 정확하다면 True, 부정확하다면 False를 선택하세요. {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}의 내용을 검토하여 {question}이 옳은지 아닌지 판단하십시오. 맞다면 True, 틀리다면 False를 선택하세요. {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}를 주의깊게 읽고, 그 내용으로부터 {question}가 참인지를 판단하십시오. 답안으로 선택할 수 있는 항목들은 {options}입니다.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}의 정보를 바탕으로, {question}에 대해 참인지 아닌지를 결정하고, 가능한 답변 {options} 중 하나를 선택하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '<보기>\n{paragraph}\n다음 문장은 참인가 거짓인가?\n{question}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '<문단>\n{paragraph}\n다음 주장이 사실인지 거짓인지 확인해주세요.\n{question}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}에 수록된 정보를 활용하여 {question}이 정확한지 검증하고, 제공된 {options} 중에서 가장 적절한 답변을 찾아 주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}의 내용을 바탕으로 {question}이 올바른지 검토한 후, 최적의 답변을 {options} 중에서 선택해 주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {paragraph}를 유심히 읽고, 이에 대한 {question}를 해결하는 가장 정확한 대답을 고르세요. {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph} 문단을 주의 깊게 살펴보고, 제시된 {question}에 가장 적절한 답변을 선택하세요. {options}',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # paragraph, question, options, answer
    # answer = {참, 거짓}
    "hyundai_nonhuman_kobest_boolq": [
        {
            'instruction': '{paragraph}의 내용을 신중하게 이해하고, 이를 바탕으로 {question}에 대해 가장 적절한 답변을 구하십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}의 내용을 주의 깊게 검토하고, 그 정보를 기반으로 {question}에 최선의 답을 제시하십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {paragraph}를 바탕으로 제시된 {question}에 대한 가장 적절한 답변을 선택하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제시된 {question}을 이해하고, {paragraph}에서 관련 정보를 찾아 가장 적절한 대답을 선택해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 문단 {paragraph}에서 주어진 질문 {question}에 대한 답변을 찾으세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}의 내용을 살펴보며 {question}에 대한 해답을 찾아내 주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}의 주제와 관련 내용을 상세히 읽고 이해한 후, 그 정보를 사용하여 {question}에 가장 합리적이고 적절한 답변을 제공해주세요',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}에 대해 정확하게 이해하고, 그 정보를 활용하여 {question}에 가장 정확한 답을 제시해주세요',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}의 내용을 철저히 읽어보고, 이를 바탕으로 {question}에 대한 답을 도출해보세요. 그 후에 {options} 중에서 가장 적절한 선택을 하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}를 읽고 이를 바탕으로 {question}에 대한 답을 찾아보세요. 이후에는 {options}로부터 정답을 선택하십시오.',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # paragraph, question, options, answer
    # answer = {참, 거짓}
    "hyundai_human_agg_kobest_boolq": [
        {
            'instruction': '{paragraph}\n윗글로 미루어볼 때 다음 문장은 참인가 거짓인가?\n{question}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}\n가 주어졌을 때\n{question}\n을 판단해주세요. \n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{paragraph}"는 "{question}"을 함의한다. 진위 여부를 판별하면? \n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{options}\n{paragraph}\n안에는\n{question}\n라는 내용이 들어가 있다. 참 또는 거짓으로 대답해.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '<보기>\n{paragraph}\n다음 문장은 참인가 거짓인가?\n{question}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '지문:\n{paragraph}\n질문:\n{question}\n{options}\n정답:',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}\n"{question}"는 참인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '질문에 답하되, 지문에 근거하여 판단하세요.\n{question}은 옳은가?\n{paragraph}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 글\n{paragraph}\n을 보고 생각했을 때,\n{question}\n은 참이니?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '참, 거짓 여부 판별\n\n근거: {paragraph}\n주장 또는 질문: {question}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}를 참조하여 {question}가 정확한지 여부를 판단하세요. 정확하다면 True, 부정확하다면 False를 선택하세요. {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}의 내용을 검토하여 {question}이 옳은지 아닌지 판단하십시오. 맞다면 True, 틀리다면 False를 선택하세요. {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}를 주의깊게 읽고, 그 내용으로부터 {question}가 참인지를 판단하십시오. 답안으로 선택할 수 있는 항목들은 {options}입니다.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}의 정보를 바탕으로, {question}에 대해 참인지 아닌지를 결정하고, 가능한 답변 {options} 중 하나를 선택하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '<보기>\n{paragraph}\n다음 문장은 참인가 거짓인가?\n{question}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '<문단>\n{paragraph}\n다음 주장이 사실인지 거짓인지 확인해주세요.\n{question}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}에 수록된 정보를 활용하여 {question}이 정확한지 검증하고, 제공된 {options} 중에서 가장 적절한 답변을 찾아 주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}의 내용을 바탕으로 {question}이 올바른지 검토한 후, 최적의 답변을 {options} 중에서 선택해 주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {paragraph}를 유심히 읽고, 이에 대한 {question}를 해결하는 가장 정확한 대답을 고르세요. {options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph} 문단을 주의 깊게 살펴보고, 제시된 {question}에 가장 적절한 답변을 선택하세요. {options}',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # paragraph, question, options, answer
    # answer = {참, 거짓}
    "hyundai_nonhuman_agg_kobest_boolq": [
        {
            'instruction': '{paragraph}의 내용을 신중하게 이해하고, 이를 바탕으로 {question}에 대해 가장 적절한 답변을 구하십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}의 내용을 주의 깊게 검토하고, 그 정보를 기반으로 {question}에 최선의 답을 제시하십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {paragraph}를 바탕으로 제시된 {question}에 대한 가장 적절한 답변을 선택하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제시된 {question}을 이해하고, {paragraph}에서 관련 정보를 찾아 가장 적절한 대답을 선택해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 문단 {paragraph}에서 주어진 질문 {question}에 대한 답변을 찾으세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}의 내용을 살펴보며 {question}에 대한 해답을 찾아내 주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}의 주제와 관련 내용을 상세히 읽고 이해한 후, 그 정보를 사용하여 {question}에 가장 합리적이고 적절한 답변을 제공해주세요',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}에 대해 정확하게 이해하고, 그 정보를 활용하여 {question}에 가장 정확한 답을 제시해주세요',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}의 내용을 철저히 읽어보고, 이를 바탕으로 {question}에 대한 답을 도출해보세요. 그 후에 {options} 중에서 가장 적절한 선택을 하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}를 읽고 이를 바탕으로 {question}에 대한 답을 찾아보세요. 이후에는 {options}로부터 정답을 선택하십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}\n윗글로 미루어볼 때 다음 문장은 참인가 거짓인가?\n{question}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}\n가 주어졌을 때\n{question}\n을 판단해주세요. \n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{paragraph}"는 "{question}"을 함의한다. 진위 여부를 판별하면? \n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{options}\n{paragraph}\n안에는\n{question}\n라는 내용이 들어가 있다. 참 또는 거짓으로 대답해.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '<보기>\n{paragraph}\n다음 문장은 참인가 거짓인가?\n{question}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '지문:\n{paragraph}\n질문:\n{question}\n{options}\n정답:',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{paragraph}\n"{question}"는 참인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '질문에 답하되, 지문에 근거하여 판단하세요.\n{question}은 옳은가?\n{paragraph}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 글\n{paragraph}\n을 보고 생각했을 때,\n{question}\n은 참이니?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '참, 거짓 여부 판별\n\n근거: {paragraph}\n주장 또는 질문: {question}\n{options}',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # paragraph, question, options, answer
    # answer = {참, 거짓}
    "hyundai_oneprompt_kobest_boolq": [
        {
            'instruction': '{paragraph}\n\n{question}\n\n{options}',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # sentence, options, answer
    # options = "선택지:\n - 긍정\n - 부정"
    # answer = {긍정, 부정}
    "kobest_sentineg": [
        {
            'instruction': '다음 문장의 감정을 긍정 또는 부정으로 분류해 줘.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}\n주어진 리뷰는 긍정적인가, 부정적인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}\n위 리뷰를 보고 감정을 분석하면?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '아래 리뷰를 긍정, 부정으로 분석해봐.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '당신은 감성 분석하는 로봇입니다. 아래 문장의 분석 결과는?\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '화자의 감정 상태를 파악하세요.\n발화: {sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 쇼핑몰 후기의 감성을 분석해 줘.\n후기: {sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}\n이 문장이 상품에 대해 어떻게 생각하고 있는 것 같니?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '쇼핑몰 후기 아무거나 한 문장 생성',
            'input': '',
            'output': '{sentence}'
        },
        {
            'instruction': '{answer}적인 상품 리뷰를 짧게 하나 써 줘.',
            'input': '',
            'output': '{sentence}'
        },
    ],
    # required keys::
    # sentence, options, answer
    # options = "선택지:\n - 긍정\n - 부정"
    # answer = {긍정, 부정}
    "hyundai_human_kobest_sentineg": [
        {
            'instruction': '가상의 구매자로서, 구매한 아이템에 대한 리뷰를 한 문장으로 작성해주세요',
            'input': '',
            'output': '{sentence}'
        },
        {
            'instruction': '인터넷 쇼핑몰에서 샀던 제품에 대한 의견을 요약하여 한 문장으로 제시해주세요',
            'input': '',
            'output': '{sentence}'
        },
        {
            'instruction': '다음 문장의 감정을 긍정 또는 부정으로 분류해 줘.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}\n주어진 리뷰는 긍정적인가, 부정적인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 문장이 긍정적인 감정을 표현하는지, 아니면 부정적인 감정을 표현하는지 판단해주세요.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '문장에서 전달되는 감정이 긍정인지 부정인지를 구별해주세요.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '당신은 감정 분석을 합니다. 주어진 문장에 대한 분석 결과를 말해주십시오.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '감정 분석을 담당하고 있습니다. 이 문장의 분석 결과를 알려주세요.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 구매자의 리뷰 내용에서 느껴지는 감정을 평가하고, 이에 해당하는 선택지를 고르세요.\n리뷰: {sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '사용자가 작성한 후기에서 드러나는 감정을 판단하고, 해당하는 답변을 선택하세요.\n후기: {sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # sentence, options, answer
    # options = "선택지:\n - 긍정\n - 부정"
    # answer = {긍정, 부정}
    "hyundai_nonhuman_kobest_sentineg": [
        {
            'instruction': '{sentence}가 포함하고 있는 감정의 성향을 객관적으로 분석하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {sentence}에 대한 객관적인 감성 분석을 실시해주세요.',
            'input': '',
            'output': '객관적인 분석을 통해, {sentence}은 {answer}의 감정을 표현하고 있음을 확인했습니다.'
        },
        {
            'instruction': '감정 분석을 위해 주어진 문장의 감정 상태를 분석해주세요. 가능한 감정 상태는 {options} 중에서 선택하실 수 있습니다.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 문장의 감정을 분석하고, 그 결과를 {options} 중에서 골라주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제시된 문장 {sentence}이 전달하는 감정을 분석하고 그 결과를 응답으로 보내주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '분석하려는 문장을 {sentence}에 입력하고, 주어진 선택지를 {options}에 맞게 넣어 sentiment analysis를 수행하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}에 대상 텍스트를 넣고, {options}에 따라 올바르게 감정 분석을 실행해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 {sentence}이 표현하는 감정을 분석하여 답변해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}의 감성 상태를 분류하고 결과를 출력하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제공된 문장 {sentence}에서 나타나는 감정의 범주를 판별하고, 그 결론을 알려주세요.',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # sentence, options, answer
    # options = "선택지:\n - 긍정\n - 부정"
    # answer = {긍정, 부정}
    "hyundai_human_agg_kobest_sentineg": [
        {
            'instruction': '가상의 구매자로서, 구매한 아이템에 대한 리뷰를 한 문장으로 작성해주세요',
            'input': '',
            'output': '{sentence}'
        },
        {
            'instruction': '인터넷 쇼핑몰에서 샀던 제품에 대한 의견을 요약하여 한 문장으로 제시해주세요',
            'input': '',
            'output': '{sentence}'
        },
        {
            'instruction': '다음 문장의 감정을 긍정 또는 부정으로 분류해 줘.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}\n주어진 리뷰는 긍정적인가, 부정적인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 문장이 긍정적인 감정을 표현하는지, 아니면 부정적인 감정을 표현하는지 판단해주세요.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '문장에서 전달되는 감정이 긍정인지 부정인지를 구별해주세요.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '당신은 감정 분석을 합니다. 주어진 문장에 대한 분석 결과를 말해주십시오.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '감정 분석을 담당하고 있습니다. 이 문장의 분석 결과를 알려주세요.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 구매자의 리뷰 내용에서 느껴지는 감정을 평가하고, 이에 해당하는 선택지를 고르세요.\n리뷰: {sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '사용자가 작성한 후기에서 드러나는 감정을 판단하고, 해당하는 답변을 선택하세요.\n후기: {sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 문장의 감정을 긍정 또는 부정으로 분류해 줘.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}\n주어진 리뷰는 긍정적인가, 부정적인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}\n위 리뷰를 보고 감정을 분석하면?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '아래 리뷰를 긍정, 부정으로 분석해봐.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '당신은 감성 분석하는 로봇입니다. 아래 문장의 분석 결과는?\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '화자의 감정 상태를 파악하세요.\n발화: {sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 쇼핑몰 후기의 감성을 분석해 줘.\n후기: {sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}\n이 문장이 상품에 대해 어떻게 생각하고 있는 것 같니?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '쇼핑몰 후기 아무거나 한 문장 생성',
            'input': '',
            'output': '{sentence}'
        },
        {
            'instruction': '{answer}적인 상품 리뷰를 짧게 하나 써 줘.',
            'input': '',
            'output': '{sentence}'
        },
    ],
    # required keys::
    # sentence, options, answer
    # options = "선택지:\n - 긍정\n - 부정"
    # answer = {긍정, 부정}
    "hyundai_nonhuman_agg_kobest_sentineg": [
        {
            'instruction': '{sentence}가 포함하고 있는 감정의 성향을 객관적으로 분석하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {sentence}에 대한 객관적인 감성 분석을 실시해주세요.',
            'input': '',
            'output': '객관적인 분석을 통해, {sentence}은 {answer}의 감정을 표현하고 있음을 확인했습니다.'
        },
        {
            'instruction': '감정 분석을 위해 주어진 문장의 감정 상태를 분석해주세요. 가능한 감정 상태는 {options} 중에서 선택하실 수 있습니다.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 문장의 감정을 분석하고, 그 결과를 {options} 중에서 골라주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제시된 문장 {sentence}이 전달하는 감정을 분석하고 그 결과를 응답으로 보내주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '분석하려는 문장을 {sentence}에 입력하고, 주어진 선택지를 {options}에 맞게 넣어 sentiment analysis를 수행하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}에 대상 텍스트를 넣고, {options}에 따라 올바르게 감정 분석을 실행해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 {sentence}이 표현하는 감정을 분석하여 답변해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}의 감성 상태를 분류하고 결과를 출력하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제공된 문장 {sentence}에서 나타나는 감정의 범주를 판별하고, 그 결론을 알려주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 문장의 감정을 긍정 또는 부정으로 분류해 줘.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}\n주어진 리뷰는 긍정적인가, 부정적인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}\n위 리뷰를 보고 감정을 분석하면?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '아래 리뷰를 긍정, 부정으로 분석해봐.\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '당신은 감성 분석하는 로봇입니다. 아래 문장의 분석 결과는?\n{sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '화자의 감정 상태를 파악하세요.\n발화: {sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 쇼핑몰 후기의 감성을 분석해 줘.\n후기: {sentence}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{sentence}\n이 문장이 상품에 대해 어떻게 생각하고 있는 것 같니?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '쇼핑몰 후기 아무거나 한 문장 생성',
            'input': '',
            'output': '{sentence}'
        },
        {
            'instruction': '{answer}적인 상품 리뷰를 짧게 하나 써 줘.',
            'input': '',
            'output': '{sentence}'
        },
    ],
    # required keys::
    # sentence, options, answer
    # options = "선택지:\n - 긍정\n - 부정"
    # answer = {긍정, 부정}
    "hyundai_oneprompt_kobest_sentineg": [
        {
            'instruction': '{sentence}\n\n{options}\n\n감정:',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # word, eun_or_neun, context_1, context_2, options, answer
    # options = "선택지\n - 같은 뜻입니다.\n - 다른 뜻입니다."
    # answer = {다른 뜻입니다., 같은 뜻입니다.}
    "kobest_wic": [
        {
            'instruction': '"{context_1}"\n{context_2}\n두 문장에서 {word}{eun_or_neun} 같은 뜻인가, 다른 뜻인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 두 문맥에서 {word}{eun_or_neun} 같은 뜻으로 쓰였는지 알려주세요.\n"{context_1}"\n"{context_2}"\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': "'{context_1}' 그리고 '{context_2}'에서 {word}{eun_or_neun} 동일한 뜻으로 사용되었는지 판단하면?\n{options}",
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '1: {context_1}\n2: {context_2}\n1과 2에서 {word}{eun_or_neun} 같은 뜻으로 쓰였어?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '"{context_1}"\n{context_2}\n두 문장에서 쓰인 {word}{eun_or_neun} 같은 뜻으로 쓰였나요, 아니면 다른 뜻으로 쓰였나요?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 문맥\n(1) {context_1}\n(2) {context_2}\n에서 {word}{eun_or_neun} 같은 뜻이니?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '단어 {word}{eun_or_neun} 다음 두 문맥에서 같은 뜻으로 쓰였는지 구분해 봐.\n1. {context_1}\n2. {context_2}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{word}{eun_or_neun} 같은 뜻으로 쓰였습니까?\n문장 1: {context_1}\n문장 2: {context_2}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '문맥 두 개가 주어진다. 단어 {word}{eun_or_neun} 같은 뜻으로 쓰였는지 판단하시오. \n{context_1}\n{context_2}\n\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{word}{eun_or_neun} 여러 뜻을 가진다.\n문장 1: {context_1}\n문장 2: {context_2}\n\n문장 1과 2에서 {word}{eun_or_neun} 같은 뜻이에요?\n{options}',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # word, eun_or_neun, context_1, context_2, options, answer
    # options = "선택지\n - 같은 뜻입니다.\n - 다른 뜻입니다."
    # answer = {다른 뜻입니다., 같은 뜻입니다.}
    "hyundai_oneprompt_kobest_wic": [
        {
            'instruction': '"{context_1}"\n{context_2}\n\n{word}\n두 문맥에서 동일한 의미인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # sentence1, sentence2
    "klue_sts": [
        {
            'instruction': '{sentence1}\n위 문장을 비슷한 말로 바꾸어 주세요.\n',
            'input': '',
            'output': '{sentence2}'
        },
        {
            'instruction': '{sentence1}\n이 말을 다른 문장으로 써 줘.\n',
            'input': '',
            'output': '{sentence2}'
        },
        {
            'instruction': '{sentence1}\n이 말을 다른 말로 다시 쓰면?\n',
            'input': '',
            'output': '{sentence2}'
        },
        {
            'instruction': '{sentence1}\n\n주어진 문장을 같은 뜻을 가진 다른 문장으로 바꾸시오.',
            'input': '',
            'output': '{sentence2}'
        },
        {
            'instruction': '{sentence1}\n를 같은 의미지만 비슷하게 말해봐.\n',
            'input': '',
            'output': '{sentence2}'
        },
        {
            'instruction': '아래 문장을 비슷하게 다시 바꿔보세요.\n\n{sentence1}\n',
            'input': '',
            'output': '{sentence2}'
        },
        {
            'instruction': '당신은 주어진 문장을 같은 의미이지만 다른 말로 바꾸어 말하는 기계입니다.\n\n문장:\n{sentence1}\n',
            'input': '',
            'output': '{sentence2}'
        },
        {
            'instruction': '다음 문장을 비슷하게 다시 표현하면?\n\n{sentence1}\n',
            'input': '',
            'output': '{sentence2}'
        },
        {
            'instruction': '다음에 오는 말을 살짝 다르게 다시 쓰세요.\n\n{sentence1}\n',
            'input': '',
            'output': '{sentence2}'
        },
        {
            'instruction': '다음에 주어지는 문장을 문맥에 맞게 다시 표현해서 말해봐.\n\n{sentence1}\n',
            'input': '',
            'output': '{sentence2}'
        },
    ],
    # required keys::
    # title, context, question, answer
    "klue_mrc": [
        {
            'instruction': '{context}\n\n문제: {question}\n',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}\n\n문제를 읽고 다음의 질문에 답하시오.\n {question}\n',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}\n\n\n {question}\n',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음 글을 읽고 질문에 답하면? \n{context}\n\n\n{question}\n',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '당신은 주어진 기사 또는 내용을 읽고 질문에 대답하는 로봇이다. 지문은 다음과 같이 주어진다.=====\n{context}\n=====\n{question}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 내용에 비추어 볼 때, {question}\n\n{context}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{question}\n\n지문: {context} \n',
            'input': '',
            'output': '{answer}'
        },
        # ==== turn around template ====
        {
            'instruction': '{context}\n\n지문에 대한 알맞은 질문을 생성하라.\n',
            'input': '',
            'output': '{question}'
        },
        {
            'instruction': '{context}\n\n이 내용에 대한 적당한 제목을 만들어.\n',
            'input': '',
            'output': '{question}'
        },
        {
            'instruction': '다음 내용에 알맞은 제목을 지어 줘.\n{context}',
            'input': '',
            'output': '{title}'
        },
    ],
    # required keys::
    # premise, hypothesis, options, answer
    # options = "선택지:\n - 수반"
    "klue_nli": [
        {
            'instruction': '전제: {premise}, 가정: {hypothesis}일 때, "전제"문장은 "가설"문장의 관계를 말하면?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음과 같이 전제 문장이 주어진다. \n"{premise}"\n\n그리고 가설 문장은 "{hypothesis}"\n\n일 때 전제는 가설을 수반하는가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '문장 1: {premise}\n문장 2: {hypothesis}\n문장 1은 문장 2를 수반하는가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '문장 A: "{premise}"\n\n문장 B: "{hypothesis}"\n일 때, 문장 A와 문장 B의 관계는?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{premise}\n에 미루어 볼 때, 문장\n{hypothesis}\n가 수반된다고 볼 수 있는가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '만약 "{premise}"\n가 성립한다면,\n""{hypothesis}"\n가 성립한다고 볼 수 있는가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '추론해보자. 만약 "{premise}"\n가 제시되면,\n"{hypothesis}"\n는 수반되는 결과일까?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '당신은 추론 능력이 있는 AI이다. 다음 두 문장은 어떤 관계인지 맞춰보자.\n\n"{premise}"\n"{hypothesis}"\n\n{options}',
            'input': '',
            'output': '{answer}'
        },
        # ==== turn around template ====
        {
            'instruction': "'{premise}'\n문장과 {answer} 관계에 있는 문장을 하나 생성하세요.\n",
            'input': '',
            'output': '{hypothesis}'
        },
        {
            'instruction': '다음 문장과 {answer} 관계의 문장을 써 줘.\n\n{premise}',
            'input': '',
            'output': '{hypothesis}'
        },
    ],
    # required keys::
    # title, options, answer
    # options = "선택지:\n - IT과학\n - 경제\n - 사회\n - 생활문화\n - 세계\n - 스포츠\n - 정치"
    # answers(example) = "경제"
    "klue_ynat": [
        {
            'instruction': '다음 기사 제목을 보고 카테고리를 분류해줘\n {title}\n{options} ',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '다음은 뉴스 기사의 제목이다. 이 기사의 유형을 분류한다면 다음 중 무엇인가?\n제목: {title}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '너는 기사 제목을 유형에 맞게 분류하는 AI이다. 다음 제목을 보고 분류하시오.\n제목: {title}\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '\n헤드라인: {title}\n이 뉴스 기사는 어떻게 분류되어야 할까?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{title}\n이 제목은 어떤 섹션의 뉴스인가?\n{options}',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '뉴스 기사 제목: {title}\n\n{options}\n\n유형 분류 결과:',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '당신은 뉴스 기사의 제목을 보고 어떤 섹션으로 분류해야 할 지 판단하는 일을 하고 있다.\n\n제목: {title}\n\n{options}\n\n분류결과:',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{options}\n\n중에서\n\n{title}\n위 제목을 분류한 결과는?',
            'input': '',
            'output': '{answer}'
        },
        # ==== turn around template ====
        {
            'instruction': '{answer} 유형의 뉴스 제목을 하나 뽑아줘\n',
            'input': '',
            'output': '{title}'
        },
        {
            'instruction': '뉴스 기사의 제목을 하나 생성해\n',
            'input': '',
            'output': '{title}'
        }
    ],
    # required keys::
    # title, options, answer
    # answers(example) = ""
    "ko_arc": [
        {
            'instruction': '{query}',
            'input': '',
            'output': '{response}'
        },
        {  # duplication for increasing probability to be selected
            'instruction': '{query}',
            'input': '',
            'output': '{response}'
        },
        {
            'instruction': '\n{query}',
            'input': '',
            'output': '{response}'
        },
        {
            'instruction': '{query}\n',
            'input': '',
            'output': '{response}'
        },
        {
            'instruction': '{query}\n\n',
            'input': '',
            'output': '{response}'
        },
    ],
    # required keys::
    # title, options, answer
    # answers(example) = ""
    "ko_commongenv2": [
        {
            'instruction': '{query}',
            'input': '',
            'output': '{response}'
        },
        {
            'instruction': '{query}',
            'input': '',
            'output': '{response}'
        },
        {
            'instruction': '{query}\n',
            'input': '',
            'output': '{response}'
        },
    ],
    # required keys::
    # title, options, answer
    # answers(example) = ""
    "ko_mmlu": [
        {
            'instruction': '{input}\nA: {A}\nB: {B}\nC: {C}\nD: {D}',
            'input': '',
            'output': '{target}. {gold}'
        },
        {
            'instruction': '{input}\n\nA: {A}\nB: {B}\nC: {C}\nD: {D}\n',
            'input': '',
            'output': '{gold}'
        },
        {
            'instruction': '{input}\n\nA: {A}\nB: {B}\nC: {C}\nD: {D}',
            'input': '',
            'output': '{gold}'
        },
        {
            'instruction': '\n{input}\n\n\nA:\n{A}\nB:\n{B}\nC:\n{C}\nD:\n{D}',
            'input': '',
            'output': '{gold}'
        },
        {
            'instruction': '{input}',
            'input': '',
            'output': '{gold}'
        },
        {
            'instruction': '{input}',
            'input': '',
            'output': '{gold}'
        },
    ],
    # required keys::
    # title, options, answer
    # answers(example) = ""
    "ko_truthfulqa": [
        {
            'instruction': '{query}',
            'input': '',
            'output': '{response}'
        },
        {  # duplication for increasing probability to be selected
            'instruction': '{query}',
            'input': '',
            'output': '{response}'
        },
        {
            'instruction': '\n{query}\n',
            'input': '',
            'output': '{response}'
        },
        {
            'instruction': '{query}\n',
            'input': '',
            'output': '{response}'
        },
        {
            'instruction': '{query}\n\n',
            'input': '',
            'output': '{response}'
        },
    ],
    # required keys::
    # title, context, question, answer
    "korquad_v1": [
        {
            'instruction': 'Title: {title}\n\nBackground: {context}\n\nQuestion: {question}\n\nAnswer:',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # title, context, question, answer
    "hyundai_human_korquad_v1": [
        {
            'instruction': 'Title: {title}\n\nBackground: {context}\n\nQuestion: {question}\n\nAnswer:',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # title, context, question, answer
    "hyundai_nonhuman_korquad_v1": [
        {
            'instruction': '바탕으로 제공된 {context}에서 {question}에 대한 답변을 생성해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': 'uc81c목 {title}을 가진 문서를 바탕으로 {question}에 대한 답변을 찾아주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {context}에서 {question}에 대한 답변을 찾아서 작성해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제공된 {context} 내용을 토대로 {question}에 대한 응답을 찾아서 기록해 주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제목 "{title}"과 관련된 질문을 바탕으로 답변을 생성해낼 수 있는 독해를 능력을 갖춘 언어 모델을 학습시켜 주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '문장 "{context}"을 기반으로 "{question}"에 응답하는데 필요한 내용을 파악한 후 답변을 작성하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {context}에서 {question}에 대한 답을 찾아주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context} 속에서 {question}의 해답을 추출해 주십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}를 토대로 {question}에 대한 가장 적절한 답변을 구성해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}를 바탕으로 {question}에 대한 최적의 응답을 작성하세요.',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # title, context, question, answer
    "hyundai_human_agg_korquad_v1": [
        {
            'instruction': 'Title: {title}\n\nBackground: {context}\n\nQuestion: {question}\n\nAnswer:',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주제: {title}\n\n배경: {context}\n\n질의: {question}\n\n응답:',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # title, context, question, answer
    "hyundai_nonhuman_agg_korquad_v1": [
        {
            'instruction': '바탕으로 제공된 {context}에서 {question}에 대한 답변을 생성해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': 'uc81c목 {title}을 가진 문서를 바탕으로 {question}에 대한 답변을 찾아주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {context}에서 {question}에 대한 답변을 찾아서 작성해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제공된 {context} 내용을 토대로 {question}에 대한 응답을 찾아서 기록해 주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '제목 "{title}"과 관련된 질문을 바탕으로 답변을 생성해낼 수 있는 독해를 능력을 갖춘 언어 모델을 학습시켜 주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '문장 "{context}"을 기반으로 "{question}"에 응답하는데 필요한 내용을 파악한 후 답변을 작성하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '주어진 {context}에서 {question}에 대한 답을 찾아주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context} 속에서 {question}의 해답을 추출해 주십시오.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}를 토대로 {question}에 대한 가장 적절한 답변을 구성해주세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': '{context}를 바탕으로 {question}에 대한 최적의 응답을 작성하세요.',
            'input': '',
            'output': '{answer}'
        },
        {
            'instruction': 'Title: {title}\n\nBackground: {context}\n\nQuestion: {question}\n\nAnswer:',
            'input': '',
            'output': '{answer}'
        },
    ],
    # required keys::
    # title, context, question, answer
    "hyundai_oneprompt_korquad_v1": [
        {
            'instruction': '제목: {context}\n\n질문: {question}\n\n응답:',
            'input': '',
            'output': '{answer}'
        },
    ]
}


# ===== processor functions for each sub-datasets =====


def _make_options_str(*options):
    """utility function that make {options} form easily. Returns raw string."""
    l = ['선택지:']
    for option in options:
        l.append(f' - {option}')
    return '\n'.join(l)


def _process_kobest_copa(template, **raw_data):
    data = copy.deepcopy(raw_data)
    label = data['label']
    question = data['question'].strip()
    data['options'] = _make_options_str(data['alternative_1'], data['alternative_2'])
    data['answer'] = data[f'alternative_{label + 1}']  # since label 0 means alternative_1
    if question == '원인':
        data['euro_or_ro'] = '으로'
        data['eun_or_neun'] = '은'
    elif question == '결과':
        data['euro_or_ro'] = '로'
        data['eun_or_neun'] = '는'
    else:
        raise NotImplementedError(f"unexpected raw data question: '{question}'")
    return {k: v.format_map(data) for k, v in template.items()}


def _process_hyundai_human_kobest_copa(template, **raw_data):
    return _process_kobest_copa(template, **raw_data)


def _process_hyundai_nonhuman_kobest_copa(template, **raw_data):
    return _process_kobest_copa(template, **raw_data)


def _process_hyundai_human_agg_kobest_copa(template, **raw_data):
    return _process_kobest_copa(template, **raw_data)


def _process_hyundai_nonhuman_agg_kobest_copa(template, **raw_data):
    return _process_kobest_copa(template, **raw_data)


def _process_hyundai_oneprompt_kobest_copa(template, **raw_data):
    return _process_kobest_copa(template, **raw_data)


def _process_kobest_hellaswag(template, **raw_data):
    data = copy.deepcopy(raw_data)
    label = data['label']
    data['options'] = _make_options_str(
        raw_data['ending_1'],
        raw_data['ending_2'],
        raw_data['ending_3'],
        raw_data['ending_4'],
    )
    data['answer'] = raw_data[f'ending_{label + 1}']
    return {k: v.format_map(data) for k, v in template.items()}


def _process_hyundai_human_kobest_hellaswag(template, **raw_data):
    return _process_kobest_hellaswag(template, **raw_data)


def _process_hyundai_nonhuman_kobest_hellaswag(template, **raw_data):
    return _process_kobest_hellaswag(template, **raw_data)


def _process_hyundai_human_agg_kobest_hellaswag(template, **raw_data):
    return _process_kobest_hellaswag(template, **raw_data)


def _process_hyundai_nonhuman_agg_kobest_hellaswag(template, **raw_data):
    return _process_kobest_hellaswag(template, **raw_data)


def _process_hyundai_oneprompt_kobest_hellaswag(template, **raw_data):
    return _process_kobest_hellaswag(template, **raw_data)


def _process_kobest_boolq(template, **raw_data):
    data = copy.deepcopy(raw_data)
    label = data['label']
    data['options'] = _make_options_str('거짓', '참')
    data['answer'] = ['거짓', '참'][label]
    return {k: v.format_map(data) for k, v in template.items()}


def _process_hyundai_human_kobest_boolq(template, **raw_data):
    return _process_kobest_boolq(template, **raw_data)


def _process_hyundai_nonhuman_kobest_boolq(template, **raw_data):
    return _process_kobest_boolq(template, **raw_data)


def _process_hyundai_human_agg_kobest_boolq(template, **raw_data):
    return _process_kobest_boolq(template, **raw_data)


def _process_hyundai_nonhuman_agg_kobest_boolq(template, **raw_data):
    return _process_kobest_boolq(template, **raw_data)


def _process_hyundai_oneprompt_kobest_boolq(template, **raw_data):
    return _process_kobest_boolq(template, **raw_data)


def _process_kobest_sentineg(template, **raw_data):
    data = copy.deepcopy(raw_data)
    label = data['label']
    data['options'] = _make_options_str('부정', '긍정')
    data['answer'] = ['부정', '긍정'][label]
    return {k: v.format_map(data) for k, v in template.items()}


def _process_hyundai_human_kobest_sentineg(template, **raw_data):
    return _process_kobest_sentineg(template, **raw_data)


def _process_hyundai_nonhuman_kobest_sentineg(template, **raw_data):
    return _process_kobest_sentineg(template, **raw_data)


def _process_hyundai_human_agg_kobest_sentineg(template, **raw_data):
    return _process_kobest_sentineg(template, **raw_data)


def _process_hyundai_nonhuman_agg_kobest_sentineg(template, **raw_data):
    return _process_kobest_sentineg(template, **raw_data)


def _process_hyundai_oneprompt_kobest_sentineg(template, **raw_data):
    return _process_kobest_sentineg(template, **raw_data)


def _process_kobest_wic(template, **raw_data):
    data = copy.deepcopy(raw_data)
    label = data['label']
    if label == 0:
        answer = '다른 뜻입니다.'
    elif label == 1:
        answer = '같은 뜻입니다.'
    else:
        raise NotImplementedError
    data['answer'] = answer
    data['options'] = _make_options_str('다른 뜻입니다.', '같은 뜻입니다.')
    data['eun_or_neun'] = bojosa(data['word'])
    return {k: v.format_map(data) for k, v in template.items()}


# 임시 코드
def _process_hyundai_human_kobest_wic(template, **raw_data):
    return _process_kobest_wic(template, **raw_data)


def _process_hyundai_nonhuman_kobest_wic(template, **raw_data):
    return _process_kobest_wic(template, **raw_data)


def _process_hyundai_human_agg_kobest_wic(template, **raw_data):
    return _process_kobest_wic(template, **raw_data)


def _process_hyundai_nonhuman_agg_kobest_wic(template, **raw_data):
    return _process_kobest_wic(template, **raw_data)


def _process_hyundai_oneprompt_kobest_wic(template, **raw_data):
    return _process_kobest_wic(template, **raw_data)


def _process_klue_sts(template, **raw_data):
    # append only 'real-label' score is above 2.0
    binary_label = raw_data['labels']['binary-label']
    if binary_label == 1:
        return {k: v.format_map(raw_data) for k, v in template.items()}
    elif binary_label == 0:
        return None
    else:
        raise ValueError('unexpected label')


def _process_klue_mrc(template, **raw_data):
    # title, context, question, answer
    data = copy.deepcopy(raw_data)
    answer = data['answers']['text'][0]
    data['answer'] = answer
    return {k: v.format_map(data) for k, v in template.items()}


def _process_klue_nli(template, **raw_data):
    # premise, hypothesis, options, answer
    data = copy.deepcopy(raw_data)
    options_str = ['수반', '중립', '모순']
    options = _make_options_str(*options_str)
    data['options'] = options
    data['answer'] = options_str[data['label']]
    return {k: v.format_map(data) for k, v in template.items()}


def _process_klue_ynat(template, **raw_data):
    # title, options, answer
    data = copy.deepcopy(raw_data)
    options_str = ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']
    label = data['label']
    options = _make_options_str(*options_str)
    data['options'] = options
    data['answer'] = options_str[label]
    return {k: v.format_map(data) for k, v in template.items()}


def _process_ko_arc(template, **raw_data):
    # raw_data:
    #  {'query': 'George는 손을 금방 따뜻하게 하기 위해 문지르는 중입니다. 어떤 피부 표면이 가장 많은 열을 발생시킬까요?',
    #  'response': '건조한 손바닥'}
    return {k: v.format_map(raw_data) for k, v in template.items()}


def _process_ko_mmlu(template, **raw_data):
    # raw_data:
    #   {'input': '이 질문은 다음 정보에 관련이...,
    #   'A': '...',
    #   'B': '...',
    #   'C': '...',
    #   'D': '...',
    #   'target': 'A'}
    data = copy.deepcopy(raw_data)
    data['gold'] = data[f'{data["target"]}']
    return {k: v.format_map(data) for k, v in template.items()}


def _process_ko_truthfulqa(template, **raw_data):
    # raw_data:
    # {'query':'...', 'response': '...'}
    return {k: v.format_map(raw_data) for k, v in template.items()}


def _process_ko_commongenv2(template, **raw_data):
    # raw_data:
    # {'query':'...', 'response': '...'}
    return {k: v.format_map(raw_data) for k, v in template.items()}


def _process_korquad_v1(template, **raw_data):
    # raw_data: dict
    # ['id', 'title', 'context', 'question', 'answers']
    data = copy.deepcopy(raw_data)
    data['answer'] = data['answers']['text'][0]
    return {k: v.format_map(data) for k, v in template.items()}


def _process_hyundai_human_korquad_v1(template, **raw_data):
    return _process_korquad_v1(template, **raw_data)


def _process_hyundai_nonhuman_korquad_v1(template, **raw_data):
    return _process_korquad_v1(template, **raw_data)


def _process_hyundai_human_agg_korquad_v1(template, **raw_data):
    return _process_korquad_v1(template, **raw_data)


def _process_hyundai_nonhuman_agg_korquad_v1(template, **raw_data):
    return _process_korquad_v1(template, **raw_data)


def _process_hyundai_oneprompt_korquad_v1(template, **raw_data):
    return _process_korquad_v1(template, **raw_data)


