<p align="left" width="100%">
<img src="assets/logo.png" alt="NLP Logo" style="width: 40%;">
</p>

## Update Logs

- 2023.09.27: initial commit
- 2023.09.28: remove redundants


# pKLUE : Korean Dataset to Instruction Tuning

영어 instruction 데이터셋의 번역 대신, 고품질 한국어 데이터셋을 instruction tuning에 사용하기 위한 연구입니다.  



## 개요

Instruction Tuning (IST)을 위해 만들어진 데이터셋이 아닌, 일반적인 한국어 고품질 데이터를 IST 가능한 형태로 가공합니다.
FLAN(Wei et al., Finetuned language models are zero-shot learners.)
에서 제시한 방법론을 차용하였습니다.

Huggingface datasets 형태로 반환하기 때문에 
FLAN 리포지토리에서 제공하는 API보다 쉽게 응용할 수 있습니다.

## 데이터 사용 방법
`mixture.py` 코드의 `get_mixture` 메서드를 이용하면 됩니다.  
### 활용 예시
```python
# dataset_names: instruction data로 사용할 source dataset list
# max_examples: 각 데이터셋의 최대 개수를 제한 (기본값: 3000)
# split: 'train' 또는 'test'
my_hf_dataset = get_mixture(dataset_names=['kullm_v2', 'kobest', 'klue'], max_examples=3000, split='train')
```

### 데이터셋 예시
```json lines
{'instruction': '아래 문장을 비슷하게 다시 바꿔보세요.\n\n숙소 위치는 찾기 쉽고 일반적인 한국의 반지하 숙소입니다.\n',
 'input': '',
 'output': '숙박시설의 위치는 쉽게 찾을 수 있고 한국의 대표적인 반지하 숙박시설입니다.'}
```
```json
{'instruction': '다음 글을 읽고 질문에 답하면? \n올여름 장마가 17일 제주도에서 시작됐다. 서울 등 중부지방은 예년보다 사나흘 정도 늦은 이달 말께 장마가 시작될 전망이다.17일 기상청에 따르면 제주도 남쪽 먼바다에 있는 장마전선의 영향으로 이날 제주도 산간 및 내륙지역에 호우주의보가 내려지면서 곳곳에 100㎜에 육박하는 많은 비가 내렸다. 제주의 장마는 평년보다 2~3일, 지난해보다는 하루 일찍 시작됐다. 장마는 고온다습한 북태평양 기단과 한랭 습윤한 오호츠크해 기단이 만나 형성되는 장마전선에서 내리는 비를 뜻한다.장마전선은 18일 제주도 먼 남쪽 해상으로 내려갔다가 20일께 다시 북상해 전남 남해안까지 영향을 줄 것으로 보인다. 이에 따라 20~21일 남부지방에도 예년보다 사흘 정도 장마가 일찍 찾아올 전망이다. 그러나 장마전선을 밀어올리는 북태평양 고기압 세력이 약해 서울 등 중부지방은 평년보다 사나흘가량 늦은 이달 말부터 장마가 시작될 것이라는 게 기상청의 설명이다. 장마전선은 이후 한 달가량 한반도 중남부를 오르내리며 곳곳에 비를 뿌릴 전망이다. 최근 30년간 평균치에 따르면 중부지방의 장마 시작일은 6월24~25일이었으며 장마기간은 32일, 강수일수는 17.2일이었다.기상청은 올해 장마기간의 평균 강수량이 350~400㎜로 평년과 비슷하거나 적을 것으로 내다봤다. 브라질 월드컵 한국과 러시아의 경기가 열리는 18일 오전 서울은 대체로 구름이 많이 끼지만 비는 오지 않을 것으로 예상돼 거리 응원에는 지장이 없을 전망이다.\n\n\n북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?\n',
 'input': '',
 'output': '한 달가량'}
```
```json
{'instruction': '다음 문장과 수반 관계의 문장을 써 줘.\n\n힛걸 진심 최고다 그 어떤 히어로보다 멋지다',
 'input': '',
 'output': '힛걸 진심 최고로 멋지다.'}
```
```json
{'instruction': '다음은 뉴스 기사의 제목이다. 이 기사의 유형을 분류한다면 다음 중 무엇인가?\n제목: 유튜브 내달 2일까지 크리에이터 지원 공간 운영\n선택지:\n - IT과학\n - 경제\n - 사회\n - 생활문화\n - 세계\n - 스포츠\n - 정치',
 'input': '',
 'output': '생활문화'}
```
```json
{'instruction': '하늘에 별이 보였다.\n위 사건의 원인은?\n상황: 하늘에 별이 보였다.\n선택지:\n - 환한 낮이 되었다.\n - 하늘이 깜깜해졌다.',
 'input': '',
 'output': '하늘이 깜깜해졌다.'}
```
