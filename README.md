# ⭐️🔥 방법 3 — Sentence-BERT(한국어 SBERT) + 비지도 이상탐지 (**가장 추천**)

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


1. `daangn_list_detail_with_image_score.csv` 읽기
2. 텍스트 대상 컬럼 선택 및 기본 전처리
3. Sentence-BERT로 설명 텍스트 임베딩 추출
4. SBERT 임베딩 → IsolationForest로 비지도 이상탐지
5. 0.0 ~ 1.0 사이로 정규화해서 `text_anomaly_score` 만들기
6. 설명이 거의 없는 매물에 대한 보정 규칙 적용 (선택 사항)
7. 최종 CSV에 `text_anomaly_score` 컬럼 붙여 저장
