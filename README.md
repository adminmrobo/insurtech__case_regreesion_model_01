**[한국어](#ko)** · English · 中文 · 日本語 · O'zbekcha · Русский (번역 예정)

---
<a id="ko"></a>
# AI(인공지능) x 비즈니스 #1 인슈어테크_03_회귀모델

건강보험 가입자 1,337명의 나이 · 성별 · BMI · 자녀 수 · 흡연 여부 · 지역으로 1년 의료비(청구 금액)를 맞히는 회귀 모델을 만드는 인터랙티브 교재입니다. 버튼만 눌러 따라 하고, 같은 코드를 Google Colab 에서 그대로 실행할 수 있습니다. 평균의 함정 → 꼬리를 지키는 데이터 정리 → 모델 다섯 개의 곡선 → 여유분과 비용 → 최고 모델을 규칙표로 옮기는 대리모델 → 견적 앱까지 여섯 편입니다. 앞 시리즈: 인슈어테크_01 분류모델 · 02 분류모델_실전.

- 글: 글 · 안상선 ((주)M-Robo 대표)
- 라이선스: CC BY-NC 4.0 (글·코드·그림). 데이터는 원 출처의 조건을 따릅니다.
- 공개 주소: https://adminmrobo.github.io/ai-biz-insurtech-03/ (저장소 이름 `adminmrobo/ai-biz-insurtech-03` 은 가칭 — 바꾸려면 build/kit.py 의 CFG 와 code/common.py 의 REPO_URL 을 고친 뒤 다시 빌드)
- 2부 실전편: https://adminmrobo.github.io/ai-biz-insurtech-04/

## 언어

| 언어 | 폴더 | 상태 |
|---|---|---|
| 한국어 | `/` | 원본 |
| English · 中文 · 日本語 · O'zbekcha · Русский | `en/` `zh/` `ja/` `uz/` `ru/` | 번역 예정 |

페이지 오른쪽 위 언어 상자로 고릅니다. 선택은 브라우저에 기억되고(`it3_lang`), 주소에 `?lang=xx`를 붙여도 됩니다.

## 편 목록

| 편 | 파일 | 제목 | 유형 |
|---|---|---|---|
| 1 | `1_charges_question.html` | 모두에게 평균만: 평균의 함정 | 해설형 · 확인 문제 4 |
| 2 | `2_data_cleaning.html` | 지우지 말고 지킨다: 중복 · 이상치 · 로그 | 방법형 · 확인 문제 3 |
| 3 | `3_five_models.html` | 모델 다섯, 곡선 다섯 | 애니메이션형 · 확인 문제 4 · 가이드라인 |
| 4 | `4_margin_lab.html` | 여유분과 비용 실험실 | 실험실형 · 확인 문제 3 · 실험 과제 |
| 5 | `5_surrogate_tree.html` | 최고 모델을 흉내 내는 나무 | 해설형 · 확인 문제 4 |
| 6 | `6_quote_app.html` | 기준서가 곧 견적: 의료비 견적 앱 | 정리형 · 확인 문제 4 · 가이드라인 |

처음 화면 `index.html` · 교육자용 정답 `answers.html` (#a1 … #a6) · 확인 문제 모두 22개

## 폴더

```
├── index.html · 1_… ~ 6_….html · answers.html · 404.html
├── assets/                   식구 그림(인슈어테크_01·02 와 같은 파일) · hero.webp(키 비주얼) · s{n}.webp(편 끝 장면 1440×800) · c{n}.webp(지도 카드, 장면 가운데 720×660)
├── code/                     정식 코드: common.py(1부·2부 같은 파일), f1.py ~ f6.py
│   └── easy/                 초보용 코드: e1.py ~ e6.py (def·for·if·lambda·들여쓰기 없음)
├── data/                     insurance.csv · split.csv · surrogate_tree.json
├── prompts/                  AI 튜터 프롬프트 ai_prompt_1.md ~ ai_prompt_6.md
├── README.md · LICENSE.md · .nojekyll
```

## 데이터

- 원 출처: https://raw.githubusercontent.com/fintech-data/Revolution/main/data/insurance.csv
- `data/insurance.csv` 는 위 파일을 **그대로 복사**한 것입니다. md5 `d5364d06246fb4bfa4cc7d1ee89ebd0d` (원본과 같음). 1,338행 × 7열(age, sex, bmi, children, smoker, region, charges).
- 코드(`common.load_raw`)는 저장소 안 파일 → 저장소 raw 주소 → 원 출처 주소 순서로 읽습니다. 환경변수 `INS_DATA` 로 경로를 줄 수도 있습니다.
- 정리 규칙(`common.clean`): 완전히 같은 행 1건(행 번호 581)만 뺀다 → 1,337행. 맞힐 금액(charges)은 자르거나 고치지 않는다. IQR 밖 charges 139행 · bmi 9행은 그대로 둔다. 결측 없음.
- **가상 데이터 생성 · 데이터 증강(합성 행 · 행 복제 · 정답에 잡음)을 하지 않습니다.**
- 1부 나누기: charges 10분위 층화 80:20, `random_state=42` → 학습 1,069 / 시험 268 (`data/split.csv`). 2부 검증: 같은 10분위로 층화한 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- 기준선(모두에게 학습 평균 $13,250): 1부 시험 MAE $9,151 · R² −0.0001 · 고액 상위 10% MAE $28,371.
- 재현: `cd code && python f1.py` … (저장소 루트에서 `python code/f1.py` 도 됩니다)

## 주의

이 시리즈는 교육용 설명입니다. 보험 가입 · 보험료 · 보험금 심사 · 투자 · 의료에 대한 조언이 아닙니다. 데이터는 공개 학습 데이터 원본이며, 실제 보험사의 현재 보험료 산정 · 심사 기준을 뜻하지 않습니다. 건강 정보(흡연 · BMI)는 설명용이며 의료 조언이 아닙니다. 실제 보험료 산정에서 성별 · 지역 · 나이 같은 특성의 사용은 감독 규정에 따라 금지되거나 제한될 수 있습니다.
