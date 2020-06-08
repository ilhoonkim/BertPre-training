# BertTraining
Google에서 제공하는 NLP 모델인 BERT를 통해 한글어 자연어 처리를 시도하는 과정에서 더 나은 방식을 찾고 정리하고 있습니다.

# 사전학습 파일 만들기
BERT GITHUB에서 내려받은 소스를 그대로 사용하기에는 한글 데이터 및 형태에 맞지 않는 부분이 있습니다.
따라서 BERT의 사전학습에 대해 이해가 필요했고 하이퍼파라미터나 코드에 수정이 필요할수도 있습니다.

**Create_pretraining_data.py** 파일 참조



## 사전학습과 데이터
<img src ="https://user-images.githubusercontent.com/45644085/83987708-216c3580-a97c-11ea-89e0-225f7ebae602.png" align="center">

BERT 사전학습은 크게 2가지를 수행하는 것을 목표로 학습이 된다고 한다.  
첫 번째는 **1. MLM(Masked Language Model)** 이다. 사전학습할 데이터를 토크나이징한 후 정해준 비율만큼의 토큰을 MASK 처리하고 해당 토큰을 예측합니다.  
두 번째는 **2. NSP(Next Sentence Prediction)** 이다. 연속된 2문장의 데이터를 만들어 실제로 이어지는 문장인지 아닌지를 라벨링하여 학습하고 이를 예측합니다.  

### 1. MASK 처리하기
```
python create_pretraining_data.py \
  --input_file=./data/nsmc_test.txt \
  --output_file=/data/nsmc_examples.tfrecord \
  --vocab_file=./nsmc_vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```
마스크 처리는 보통 디폴트값을 유지해서 학습했습니다.   
하이퍼파라미터에서 masked_lm_prob, max_predictions_per_seq 가 MASK 처리와 관련되어 있습니다.  
- masked_lm_prob는 각 example의 토큰 중에서 얼마만큼의 비율을 마스킹 처리할것이냐와 연관됩니다. 보통은 0.15(15%)로 설정되어 있어 크게 바꾸지 않고 사용하고 있습니다.  
- max_predictions_per_seq 는 한 example 에서 최대 마스킹될 수 있는 토큰의 갯수입니다. max_seq_length * masked_lm_prob는 라고 생각하시면 되겠습니다.

### 2. NSP 형태의 인스턴스 만들기
사전학습용 파일을 만드는데 가장 이해가 어려웠던 부분입니다.  
NSP를 어떤 형태로 만들어서 학습하는지 이해가 필요합니다.


```
교도소 이야기구먼.. 솔직히 재미는 없다. 이런 별로인 영화에는 솔직히 평점 2드립니다. 

>>> tokens: [CLS] 교도소_ 이야기 구먼_ .. 솔직히_ 재미 는_ 없다 .. [SEP] 이런 별로 인_ [MASK] 에는_ [MASK] 평점 2 드립니다_[SEP]
>>> segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1
>>> is_random_next: True
>>> masked_lm_positions: 11 15 16 22
>>> masked_lm_labels: 조정 기대했던_ 내 인
```
