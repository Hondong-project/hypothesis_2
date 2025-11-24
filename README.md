#가설2: 심리적 압박·유혹·과장 표현이 많은 매물은 전세사기일 확률이 높을 것이다. 
# ⭐️🔥 Sentence-BERT(한국어 SBERT) + 비지도 이상탐지 (**가장 추천**)

가장 높은 성능 + 자연어 의미를 제대로 반영

→ LLM 기반 embedding 사용하면 “사기 문구의 뉘앙스”도 반영됨.

## 모델 추천:

- `sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens`
- `jhgan/ko-sbert-multitask`

## 진행 단계

### ① 설명(description)을 문장 단위로 임베딩

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jhgan/ko-sbert-multitask")
emb = model.encode(df["description"].tolist())
# emb.shape = (N, 768)

```

### ② 임베딩 → 이상탐지 모델 적용

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(emb)
raw = -iso.score_samples(emb)

```

### ③ 0~1 정규화

```python
min_s, max_s = raw.min(), raw.max()
df["text_anomaly_score"] = (raw - min_s) / (max_s - min_s + 1e-9)

```

### ④ CSV에 저장

`text_anomaly_score` 컬럼 생성 완료.

**장점**

- 설명 텍스트의 의미·감정·뉘앙스 반영
- 과장/압박 문구 자동 탐지
- “비정상적 표현 패턴”을 잘 잡음
- 이미지 anomaly score 방식을 그대로 인용 가능 (일관성 ↑)

**단점**

- 모델 설치가 필요
- 설명이 거의 비어 있는 매물의 경우 score가 왜곡될 수 있음
    
    → 이 경우 “20자 미만 → score +0.2” 같은 규칙 추가하면 됨.




-----------------------------------------------------------------------------------------------------
###매물 설명 텍스트를 SBERT 임베딩 + IsolationForest로 분석해서, **각 매물별 text_anomaly_score (0.0 ~ 1.0)**를 만든다.


### 1. `daangn_list_detail_with_image_score.csv` 읽기

1. 이전 단계(가설4)에서 만든 CSV 파일을 불러온다.
    - 예: `daangn_list_detail_with_image_score.csv`
2. 여기에는 이미 `image_anomaly_score`가 들어있고,
    
    지금 여기에 `text_anomaly_score`를 추가하는 게 목표.
    

---

### 2. 텍스트 대상 컬럼 선택 및 기본 전처리

1. 매물 설명이 들어 있는 컬럼을 선택한다.
    - 예: `description`
2. 결측치(NaN)나 너무 짧은 텍스트 처리:
    - `description`이 없는 행 → `" "`(빈 문자열)로 채우거나 별도 처리
    - 길이가 **너무 짧은 설명(예: 10~20자 미만)**은 정보 부족으로 보고
        
        나중에 점수 보정할 수 있도록 따로 마스크를 만들어 둔다.
        
        - 예: `short_desc_mask = df['description'].str.len() < 20`

---

### 3. Sentence-BERT로 설명 텍스트 임베딩 추출

1. 한국어 SBERT 모델 로드:
    - 예: `jhgan/ko-sbert-multitask` 또는 `sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens`
2. `description` 컬럼 전체를 리스트로 만들어 SBERT에 넣어서 임베딩을 얻는다.
    - 각 매물 → 길이 768짜리 벡터 하나 (텍스트 의미를 숫자로 표현한 것)

> 이 단계까지 하면:
> 
> 
> **“각 매물 설명 텍스트의 의미를 벡터(숫자 배열)로 바꿈”** 이라고 이해하면 됨.
> 

---

### 4. SBERT 임베딩 → IsolationForest로 비지도 이상탐지

1. SBERT 임베딩 전체를 `X`라고 하고, 여기에 **IsolationForest**를 학습시킨다.
    - `contamination=0.05` 정도로, 상위 5%를 이상치로 간주 (원하면 조정)
2. 학습된 모델에서 각 매물에 대한 score를 얻는다:
    - `score_samples`는 값이 클수록 정상, 작을수록 이상치
        
        → 우리는 `raw = -score_samples`로 부호를 바꿔서
        
        **값이 클수록 “이상(의심)”이 되게 만든다.**
        

> 여기까지가
> 
> 
> **“설명 텍스트가 전체 패턴에서 얼마나 튀는지 수치화했다”** 단계.
> 

---

### 5. 0.0 ~ 1.0 사이로 정규화해서 `text_anomaly_score` 만들기

1. `raw` 값들 중 최소/최대값을 이용해 **min-max 정규화**를 한다.

![image.png](attachment:50c2b3fb-d672-4c8b-9d0e-0686b9be863f:image.png)

1. 이렇게 하면 모든 매물에 대해 `0.0 ~ 1.0` 범위의 값이 생김.
    - 0에 가까울수록 “텍스트가 평범함”
    - 1에 가까울수록 “텍스트가 다른 매물들과 패턴이 많이 다름(과장/압박일 가능성 ↑)”

---

### 6. 설명이 거의 없는 매물에 대한 보정 규칙 적용 (선택 사항)

SBERT + IsolationForest는 **설명이 거의 없는 매물**에 대해

의미 있는 판단을 하기 어려울 수 있다. 그래서:

1. 아까 만들어둔 `short_desc_mask`를 사용해서,
    - 설명 길이 < 20자 같은 매물에 대해
    - `text_anomaly_score`에 **+0.2** 같은 penalty를 줄 수 있다.
2. 단, 값이 1.0을 넘지 않게 `clip(0, 1)` 처리.

이 규칙에 대한 해석:

> “설명이 거의 없는 매물은 정보 비대칭이 크기 때문에,
> 
> 
> 텍스트 관점에서 더 위험하다고 본다.”
> 

---

### 7. 최종 CSV에 `text_anomaly_score` 컬럼 붙여 저장

1. 기존 `df`에 `text_anomaly_score`라는 새 컬럼을 만든다.
2. 위에서 계산한 0~1 값을 채워 넣는다.
3. 최종적으로:
    - `image_anomaly_score` (가설4)
    - `text_anomaly_score` (가설2)
    - (추후) `price_anomaly_score`, `meta_anomaly_score`
        
        가 함께 포함된 CSV를 **새 파일명**으로 저장한다.
        
    - 예: `daangn_list_detail_with_image_text_score.csv`
