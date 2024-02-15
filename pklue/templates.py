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


def _process_kobest_boolq(template, **raw_data):
    data = copy.deepcopy(raw_data)
    label = data['label']
    data['options'] = _make_options_str('거짓', '참')
    data['answer'] = ['거짓', '참'][label]
    return {k: v.format_map(data) for k, v in template.items()}


def _process_kobest_sentineg(template, **raw_data):
    data = copy.deepcopy(raw_data)
    label = data['label']
    data['options'] = _make_options_str('부정', '긍정')
    data['answer'] = ['부정', '긍정'][label]
    return {k: v.format_map(data) for k, v in template.items()}


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
