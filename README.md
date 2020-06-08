# BertTraining
Google에서 제공하는 NLP 모델인 BERT를 통해 한글어 자연어 처리를 시도하는 과정에서 더 나은 방식을 찾고 정리하고 있습니다.

## 사전학습 파일 만들기
BERT GITHUB에서 내려받은 소스를 그대로 사용하기에는 한글 데이터 및 형태에 맞지 않는 부분이 있습니다.
따라서 BERT의 사전학습에 대해 이해가 필요했고 하이퍼파라미터나 코드에 수정이 필요할수도 있습니다.

**Create_pretraining_data.py** 파일 참조



### 사전학습과 데이터
<img src ="https://user-images.githubusercontent.com/45644085/83987708-216c3580-a97c-11ea-89e0-225f7ebae602.png" align="center">

BERT 사전학습은 크게 2가지를 수행하는 것을 목표로 학습이 된다고 한다.  
첫 번째는 **MLM(Masked Language Model)** 이다. 사전학습할 데이터를 토크나이징한 후 정해준 비율만큼의 토큰을 MASK 처리하고 해당 토큰을 예측합니다.  
두 번째는 **NSP(Next Sentence Prediction)** 이다. 연속된 2문장의 데이터를 만들어 실제로 이어지는 문장인지 아닌지를 라벨링하여 학습하고 이를 예측합니다.  
