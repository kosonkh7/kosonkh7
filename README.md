## Honor to meet YOU! 
- I'm **Gwangha Go**👋, researcher with strong intellectual curiosity and prudence.
- I wanna contribute to solving realistic problem through well-organized technology.
- I have a interest in Data Science and aim to solve complex problems experimentally.

## Research Experience
***
### EEG 데이터를 이용한 알츠하이머병 진단을 위한 딥러닝 기반 분류 모델의 성능 평가 
2024.01 ~ 2024.06 개인 | 졸업논문 | **KCC 2024 수상** [[paper](https://github.com/kosonkh7/kosonkh7/blob/main/paper.pdf)] [[poster](https://github.com/kosonkh7/kosonkh7/blob/main/poster.pdf)]

#### 문제 정의
&#45; 현대 의학 기술의 발전에 따라, 치료보다 **질환의 조기 진단 및 예방이 보다 중요**하게 여겨지는 추세. <br>
&#45; 알츠하이머병 진단에 주로 사용되는 MRI는 상당한 신경 변성이 진행된 후에야 유효하고, 시간과 비용 많이 소요. <br>
&#45; **EEG(뇌파)는 상대적으로 측정 접근도가 뛰어난 생체 지표**, 조기 진단 및 예방에 효과적. <br>
&#45; EEG에 딥러닝 기반 분류 모델을 적용하여, **알츠하이머병의 조기 진단에 기여**하는 것이 목적. <br>

#### 연구 내용

&#45; **MNE**를 이용하여 **EEG에 포함된 노이즈 제거**를 위해 Band Pass Filter, Re-Referencing, 독립 성분 분석(ICA) 적용.<br>
&#45; EEG 데이터 분류에 특화된 **8개의 딥러닝 기반 분류 모델 선정**. <br>
&#45; **Optuna**를 이용하여 각 모델마다 알츠하이머병 진단에 최적화 된 **하이퍼파라미터 조합 탐색**. <br>
&#45; 최적화된 각 모델 성능 평가를 위한 **Leave-One-Subject-Out(LOSO) 검증 구현**. <br>

#### 성과 및 느낀점

&#45; **EEGConformer와 ATCNet 모델이 각각 78.59%와 76.36%의 정확도**를 기록하며 가장 높은 성능 보임. <br>
&#45; **두 모델은 트랜스포머 네트워크의 Multi-Head Attention 모듈을 통합하여 구성된 것이 공통적인 특징.** <br>
&#45; 추가 데이터 수집, 데이터 증강 기법 적용을 통해 모델의 표현력과 하이퍼파라미터의 최적화를 더욱 극대화할 수 있을 것. <br>
&#45; 향후 EEG 데이터를 활용하여 독자적인 딥러닝 기반 분류 모델을 설계할 때, 비교 평가 자료로 활용 기대. <br>

***

### Text-to-Video 모델을 활용한 공작기계 매뉴얼 영상 생성 연구
2024.03 ~ 2024.06 팀 5명 | 창의적종합설계 [[more detail](https://github.com/kosonkh7/T2V-Machine-tool-Fine-Tuning)]

#### 문제 정의
&#45; 공작기계 조작법이 담긴 매뉴얼은 방대한 내용을 담고 있어 작업자들이 숙지하기에 어려움을 겪음. <br>
&#45; Sora와 같은 Text-to-Video 모델의 발전으로 다양한 산업군에서 전방위적인 기술 혁신 예상. <br>
&#45; **알고자 하는 공작기계 공법을 동영상으로 바로 제시하여 작업자에게 편의를 제공**하는 것이 목적. <br>

#### 연구 내용
&#45; 공작기계 이미지, 동작 영상 **데이터셋 구성. 각 데이터 별 적절한 프롬프트 설계 및 라벨링**. <br>
&#45; **AnimateDIFF 프레임워크** 활용. (사전 학습된 Text-to-Image(T2I) 모델에 Motion Module을 적용하여 영상으로 변환하는 프레임워크) <br>
&#45; **LoRA**를 이용하여 T2I 모델을 공작기계 이미지 데이터로 **Fine-Tuning**. <br>
&#45; **Motion Director**를 이용하여 Motion Module을 공작기계 영상 데이터로 **Fine-Tuning**.   <br>


#### 성과 및 느낀점 [[생성 결과 영상](https://github.com/kosonkh7/T2V-Machine-tool-Fine-Tuning?tab=readme-ov-file#conclusion)]
&#45; 본 연구에선 **공작기계의 가장 핵심적인 공정 기법인 밀링과 터닝을 구현**. <br>
&#45; 추가적인 데이터 수집과 학습을 통해 머시닝 센터(복합가공기)의 모든 공정을 반영하는 모델 개발 기대. <br>
&#45; 공작기계는 제조사마다 다르기에, 특정 기계의 영상 / 이미지 데이터만을 수집 및 학습에 활용한다면 더 좋은 결과 생성할 수 있을 것. <br>
&#45; 향후 Sora와 같은 모델이 오픈소스로 공개된다면 이에 맞는 Fine-Tuning을 통해 보다 정밀한 영상 생성 가능할 것. <br>

***

### U-Net 기반 딥러닝 모델을 활용한 뇌종양 분할 및 성능 평가
2022.09 ~ 2022.12 팀 4명 | 데이터분석캡스톤디자인 [[more detail](https://github.com/kosonkh7/Encephaloma-Segmentation)]

#### 문제 정의
&#45; **뇌종양 제거 수술 및 추적 관리에는 매우 정밀한 종양 영역 검출 요구**. <br>
&#45; 영역 분할을 반복적으로 수행해야 하는 전문의 피로도 증가 문제 대두. <br>
&#45; **의료 데이터 분할에 특화된 U-Net 기반 네트워크를 최적화하여 가장 효과적인 영역 분할 모델 선정**. <br>

#### 연구 내용
&#45; **Elastic Deformation**을 이용하여 데이터 증강. <br>
&#45; **U-Net 기반 4가지 네트워크**(Unet, ResUnet, DeepResUnet, HybridResUnet).  <br>
&#45; 모델 별 학습 평가 지표 모니터링. <br>

#### 성과 및 느낀점
&#45; **UNet 모델이 0.7833(IoU), 0.8585(F1-score)로 가장 높은 성능** 보임. <br>
&#45; 보다 간결한 딥러닝 아키텍처를 가지는 모델이 더 좋은 성능을 보일 수 있음을 확인. <br>
&#45; 뇌종양 뿐만 아니라 정밀한 영역 분할을 필요로 하는 질환의 진단을 보조하는 기법으로 널리 확대될 것으로 기대. <br>

***

### 대기행렬이론 기반 컴퓨터 시뮬레이션을 활용한 공공자전거 대여 시스템 재고 불균형 해소 연구
2022.03 ~ 2022.06 팀 2명 | 컴퓨터시뮬레이션 [[more detail](https://github.com/kosonkh7/PBSS-Analysis)]

#### 문제 정의
&#45; **가까운 대여소에 따릉이가 없어서 이용하지 못한 불편함을 해소하는 것이 목적**.  <br>
&#45; 공공 자전거 대여 시스템은 **대여소(서버) 내의 평균 자전거 수(L)와, 대여소 내 평균 자전거 거치 시간(W) 낮을수록 효율적.** <br>
&#45; **L 값이 0이 되지 않으면서, 모든 대여소의 L값의 분산을 최소화하는 적절한 대안 모색.** <br>
&#45; 유동 인구가 가장 많은 잠실역 인근 30개 대여소 이력 데이터를 바탕으로 분석을 진행. <br>

#### 연구 내용
&#45; Pandas, Matplotlib, Folium을 활용한 EDA, 재고 불균형 심한 대여소 조사, 데이터 이상치 제거. <br>
&#45; **@Risk**를 이용하여 각 대여소의 대여, 반납 데이터 별로 **방문 분포 및 모수 통계적 추정**. <br>
&#45; 재고 불균형이 심한 6개 대여소를 대상으로 14개의 **재분배 조정 가설 시나리오 설정**. <br>
&#45; **Simio**를 이용하여 자전거 대여, 반납 및 분배 시스템 설계 및 시뮬레이션. <br>
&#45; 각 시나리오 별 결과 L의 평균과 분산의 감소에 대한 **T-Test, F-Test를 시행하여 유의성 검사**. <br>

#### 성과 및 느낀점
&#45; T-Test 결과 14개의 시나리오 모두 p-value가 0.05 이하로, **전부 L의 평균 차이가 유의미함을 확인**. <br>
&#45; F-Test 결과 2개의 시나리오만 p-value가 0.05 이하로, **L의 분산 또한 감소한 2개의 시나리오를 최적으로 선정**. <br>
&#45; **사용량이 매우 크거나 적은 대여소를 우선적으로 재배치 하는 것이, 기존 시스템 대비 공공 자전거 이용 효율을 증가 시키는 것을 확인**. <br>
&#45; 활용 프로그램, 컴퓨팅 자원 제약 없이 더 다양한 시나리오를 실험해보면 보다 최적화된 재배치 운영 전략 탐색 가능할 것. <br>


***

## Educations
- **B.E.** in Department of **Industrial & Management Systems Engineering**, Kyung Hee University, 2017.03 - 2024.08.
- **B.E.** in Department of **Software Convergence**, Kyung Hee University, 2017.03 - 2024.08.
- KT AIVLE School AI Developer Track, Software Engineering Bootcamp, 2024.09 - Present(2025.02)
- GPA: 3.84 / 4.5

## Certifications
- AICE Associate (KT, 2025.02)
- 빅데이터분석기사 (과학기술정보통신부, 통계청, 2023.12) [[repo](https://github.com/kosonkh7/Data_Analysis_Portfolio/tree/main/BigDataAnalysis_Certificate)]
- SQLD (한국데이터산업진흥원, 2021.10)
- ADsP (한국데이터산업진흥원, 2020.09)
- ITQ OA Master (한국생산성본부, 2019.12)
- 컴퓨터활용능력(2) (대한상공회의소, 2019.08)

## Awards
- 한국컴퓨터종합학술대회(KCC) 학부생 부문 장려상 (한국정보과학회, 2024)
- 지식재산능력시험 우수상 (-대학교 지식창업교육센터, 2018)
- FIELD(Future Industrial Engineering Leaders and Dreamers) 컴페티션 대상 (대한산업공학회, 2017)

## Language skills
- TOEIC(855)

## Extra Activities
- 생성형 인공지능 기반 AI 커버곡 유튜브 채널 운영 [[link](https://www.youtube.com/channel/UCuizYZgtZva8zTvNwpy4Cbg)] [[repo](https://github.com/kosonkh7/RVC_Voice_Conversion)]
- SQL 데이터 분석 캠프 이수 (데이터리안, 온라인 과정, 2023.11~2023.12)
- Google Advanced Data Analytics Professional 자격 수료 (Coursera, Google, 온라인 과정, 2023.03~2023.06)
- 000000 IB전략부서 기업 분석 체험형 인턴 (2021.02~2021.03)

## Tech Stacks
[![Solved.ac 프로필](http://mazassumnida.wtf/api/mini/generate_badge?boj=kosonkh7)](https://solved.ac/kosonkh7) <br>
[![My Skills](https://skillicons.dev/icons?i=py,mysql,java,sklearn,tensorflow,pytorch,docker,kubernetes,github,vscode,idea,notion&theme=light)](https://skillicons.dev)

<!--
**kosonkh7/kosonkh7** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
