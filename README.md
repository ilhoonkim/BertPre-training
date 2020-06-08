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
- 마스킹은 해당 토큰을 [MASK]로 치환합니다.

### 2. NSP 형태의 인스턴스 만들기
사전학습용 파일을 만드는데 가장 이해가 어려웠던 부분입니다.  
NSP를 어떤 형태로 만들어서 학습하는지 이해가 필요합니다.


```
교도소 이야기구먼.. 솔직히 재미는 없다.    
이런 별로인 영화에는 솔직히 평점 2드립니다. 

>>> tokens: [CLS] 교도소_ 이야기 [MASK] .. 솔직히_ 재미 [MASK] 없다 .. [SEP] 이런 별로 인_ [MASK] 에는_ [MASK] 평점 2 드립니다 [MASK] [SEP]
>>> segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1
>>> is_random_next: False  # 두 문장이 실제로 이어진 문장인지의 여부
>>> masked_lm_positions: 3 7 14 16   #마스킹된 토큰의 위치 
>>> masked_lm_labels: 구먼_ 는_ 영화 솔직히_  # 마스킹된 토큰
```
다음은 NSP 형태의 인스턴스를 보여드리기 위해 이미 잘 가공된 인스턴스를 보여드렸습니다. 
```
[CLS] 첫 문장 [SEP] 다음문장 [SEP]
>>> is_random_next: False  
```
NSP를 학습하기 위해서는 다음과 같은 형태로 인스턴스가 만들어져야 되는 것입니다.

그런데 문제가 있었습니다. BERT의 Create_pretraining_data.py를 그대로 사용하여 인스턴스를 만들게 되면 다음과 같이 인스턴스가 만들어지게 됩니다.

```
교도소 이야기구먼.. 솔직히 재미는 없다.    
이런 별로인 영화에는 솔직히 평점 2드립니다.
왜케 평점이 낮은건데? 꽤 볼만한데.. 헐리우드식 화려함에만 너무 길들여져 있나?
액션이 없는데도 재미 있는 몇안되는 영화
사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다
다 짤랐을꺼야. 그래서 납득할 수 없었던거야.. 그럴꺼야.. 꼭 그랬던걸꺼야..

>>> tokens: [CLS] 교도소_ 이야기 [MASK] .. 솔직히_ 재미 [MASK] 없다 .. 이런 별로 인_ 영화 [MASK] 솔직히 평점 [MASK] 드립니다 [MASK] 왜케 평점 [MASK] 낮은 건데 ? 꽤 볼만 [MASK] .. 헐리우드 식 [SEP] 다 짤랐을꺼야 . [MASK] 납득 할 [MASK] 없 었던 거야 .. [MASK] .. 꼭 그랬던 [MASK] .. [SEP]
>>> segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1
>>> is_random_next: True
```
다음을 보고 바로 이해하기는 어렵다고 생각됩니다.   
기존 사전학습용 인스턴스를 만드는 방식은 txt 파일을 불러와서 무조건 max_seq_length 만큼 잘라서 리스트를 만들고 해당 요소들을 실제로 붙어있었던 경우면 is_random_next를 False로 아니라면 True 로 주게 되는 경우입니다. 따라서 같은 문장이 아니더라도 max_seq_length만큼 붙어버리거나 혹은 길이가 길어지면 문장 중간에서 잘려버리는 경우가 생기게 됩니다.   
   
따라서 NSP를 잘 학습하기 위한 인스턴스를 만들기 위해서는 텍스트 데이터의 형태와 특성에 따라 코드를 변경해줄 필요가 있습니다.
이 부분은 정답의 코드가 없지만 코드에 맞춰 데이터를 준비하던지 혹은 데이터에 맞게 코드를 변경하거나 코드에 맞게 데이터를 맞추는 노력이 필요합니다.   

저의 경우에는 학습데이터가 다음과 같은 형태라고 가정하고 코드를 변경해보았습니다.
```
안녕하세요 반갑습니다.   
제 이름은 김일훈입니다.
저의 나이는 32살이구요.
현재 주식회사리비에서 일하고 있습니다.
회사에서 팀은 선행기술팀이구요
팀 내에서 팀장을 맡고 있어요.

안녕하세요 제 이름은 철수입니다.
제 이름이 영수인 이유는 수영을 좋아하기 때문입니다.
제 농담이 재미가 없으셨다면 죄송합니다.
```
저는 데이터를 한줄씩 읽어올것이고 각 줄은 다음 줄과 연속된 문장입니다. 다만 다른 문장인 경우에는 사이에 빈 줄을 넣을 것입니다.

그리고 제가 원하는 아웃풋의 형태는 다음과 같습니다.
```
[CLS] 안녕 하세요 반갑 [MASK] [SEP] 제 이름 [MASK] 김일훈 [MASK] .[SEP]
>>> is_random_next: False

[CLS] 안녕 하세요 반갑 [MASK] [SEP]제 [MASK] 이 영수 인 이유 [MASK] 수영 을 [MASK] 아하기 때문 [MASK] .[SEP]
>>> is_random_next: True
```

이러한 결과값을 얻기 위해서 **Create_pretraining_data.py** 의 일부분을 수정해주었습니다.
```
def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    ...
    ...
    ...
    ...
      if i == len(document) - 1 or len(current_chunk) == 2: #current_length >= target_seq_length:
    
```
기존에 max_seq_length 까지 문장을 이어붙이는 것을 고치고자 current_chunk에 2개의 segment 문장이 들어오는 순간 그만두도록 처리했습니다.

```
  ...
  ...
      a_end = 1
      #if len(current_chunk) >= 2: 
      #  a_end = rng.randint(1, len(current_chunk) - 1)
      tokens_a = []
      for j in range(a_end):
        tokens_a.extend(current_chunk[j]) 
    ...
    ...
```
tokens_a 부분에 몇개의 segment를 넣는지 결정하는 a_end를 1로 픽스하여 무조건 한 문장만 넣도록 처리했습니다.
current_chunk 안에 문장이 2개이므로 자동으로 남은 문장이 tokens_b에 들어가게 됩니다.


```
      random_document = all_documents[random_document_index]
      random_start = rng.randint(0, len(random_document) - 1)
      for j in range(random_start, len(random_document)):
        tokens_b.extend(random_document[j])
        #if len(tokens_b) >= target_b_length:
        break

```
tokens_b 의 경우에도 max_seq_length 직전까지 random 문장을 추가하므로 한 문장만 넣도록 처리하였습니다.   

```
...
...
flags.DEFINE_float(
    "short_seq_prob",1,  #0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")
```
해당 파라미터는 default 값이 0.1인데 1로 변경해주었습니다. 제가 사용할 데이터의 경우는 한 문장이 max_seq_length인 128을 넘는 경우가 하나도 없었기 때문입니다.   

- - -

다음과 같은 처리를 끝내면 처음에 기획한 형태로 사전학습 인스턴스가 만들어지게 됩니다. 다만 데이터도 기획에 맞도록 준비해야 된다는 점은 변하지 않습니다.  
**Create_pretraining_nsp_data.py** 참조

비가공 데이터가 들어와도 기획한 형태로 데이터를 바꾸는 코드를 만들어볼까도 했으나 워낙 데이터마다 형태가 달라서 우선은 생략하겠습니다.

## 사전학습 하기
지금까지는 사전학습을 하기 위해 비가공 상태의 텍스트 데이터를 사전학습용 인스턴스로 바꿔주는 일련의 과정을 확인하고 수정해보았습니다.   
사실 사전학습할 인스턴스 파일이 만들어지면 그 이후는 하이퍼파라미터의 변경 정도의 일만 남았을 뿐이라고 봐도 무방합니다.

```
python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```
다음은 BERT github에서 예시로 제공하는 사전학습 명령어입니다.   

각 argument가 의미하는 바는 다음과 같습니다.   
- input_file : 사전학습을 위해 준비한 인스턴스 파일(Create_pretraining_data.py 의 실행 결과 output_file)
- output_dir : 사전학습 모델이 저장될 경로
- do_train : 학습 여부
- do_eval : 검증 여부
- bert_config_file : BERT 학습을 위한 모델 구성 관련 json 파일
- init_checkpoint : 학습을 이어할 모델 경로
- train_bath_size : 한 번에 몇 개의 example을 학습하는지 (학습 속도와 연관)
- max_seq_length : 하나의 문장이 가질 수 있는 최대 토큰 수
- max_predictions_per_seq : 한 문장내에 예측하는 최대 토큰 수(최대 MASK 수)
- num_train_steps : 목표 학습량
- num_warmup_steps : 오버슈팅을 방지하기 위해 해당 step까지 learning_rate를 점진적으로 증가
- learning_rate : 한 번의 학습마다 얼마만큼 학습할지에 대한 여부

