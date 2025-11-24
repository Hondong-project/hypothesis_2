1. `daangn_list_detail_with_image_score.csv` 읽기
2. 텍스트 대상 컬럼 선택 및 기본 전처리
3. Sentence-BERT로 설명 텍스트 임베딩 추출
4. SBERT 임베딩 → IsolationForest로 비지도 이상탐지
5. 0.0 ~ 1.0 사이로 정규화해서 `text_anomaly_score` 만들기
6. 설명이 거의 없는 매물에 대한 보정 규칙 적용 (선택 사항)
7. 최종 CSV에 `text_anomaly_score` 컬럼 붙여 저장
