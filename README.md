# 이미지와 증상 정보를 활용한 멀티모달 AI 기반 일반피부질환 사전 선별 모델
본 프로젝트는 [SCIN Dataset](https://huggingface.co/datasets/google/scin)을 기반으로 이미지와 증상 정보를 활용하여 일반 피부질환을 사전 선별하는 멀티모달 AI 모델을 개발합니다.

모델 학습 및 평가는 **Google Colab 환경**에서 진행합니다.  
자세한 실행 방법은 아래 README에 정리했으며, 코드를 간편하게 실행할 수 있도록 별도의 [Google Colab 노트북](https://colab.research.google.com/drive/1VoOmEk4PUYlazpa8ieVkUJWWxfiu8hUy?usp=sharing)도 공유합니다.


<br>

## 1 ) 데이터셋 및 실행 환경 준비


1. **Google Drive 공유 폴더 연결**
  
   전처리된 CSV 및 이미지 데이터가 저장된 [Google Drive 공유 폴더](https://drive.google.com/drive/folders/1lNW6DgNF92QDuI56RPHqUYA0uIZX4TaO?usp=sharing)를 내 드라이브에 바로가기로 추가합니다.
   
   a. 공유 폴더 이름 클릭  
   b. **정리 → 비로가기 추가** 클릭  
   c. **모든 위치 → 내 드라이브 → 추가** 클릭  
   
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/b586aa36-b86a-4358-8197-7923b9f36590" />
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/af6731da-03df-4e69-ae19-57c8258c2a79" />
  

2. **GPU 런타임 설정**  
  
   Google Colab 상단 메뉴에서 런타임 → 런타임 유형 변경 → T4 GPU → 저장을 선택합니다.


3. **Google Drive 마운트**

```
   from google.colab import drive
   drive.mount('/content/drive')
  ```
<br> 

4. **GitHub 저장소 Clone**
```
    !git clone https://github.com/multimodal-skin-ai/data.git
    %cd data
```


<br>  <br> 
## 2 ) 모델 학습 및 평가

#### 2-0. 이미지 데이터 준비  
이미지 로딩 속도를 개선하기 위해 Google Drive에 저장된 이미지 데이터를 Colab 로컬 저장소로 복사합니다.
```
!cp -r "/content/drive/MyDrive/데이터사이언스응용 - 고양이 팀/image" /content/images
```

### 2-1. 이미지 단독 모델 학습
이미 학습된 모델이 outputs/image_best.pt에 저장되어 있으므로, 모델 학습 과정을 생략할 수 있습니다.
```
!python -m training.train \
  --data-csv "/content/drive/MyDrive/데이터사이언스응용 - 고양이 팀/integration/final_integrated_dataset.csv" \
  --image-root "/content/images" \
  --mode image \
  --class-weighting
```
학습 과정에서는 Validation Macro F1을 기준으로 최적 모델을 저장합니다.

<br>

### 2-2. 이미지 단독 모델 테스트
```
!python -m evaluation.evaluate \
    --test-csv "/content/drive/MyDrive/데이터사이언스응용 - 고양이 팀/integration/final_integrated_dataset.csv" \
    --checkpoint "outputs/image_best.pt" \
    --robustness-root "/content/images/test_images" \
    --batch-size 16 \
    --output-dir outputs/evaluation
```
원본 이미지, 해상도 저하 이미지, 블러 적용 이미지, 밝기 저하 이미지에서 각각 Accuracy와 Macro F1을 측정하고, 원본 이미지 대비 Macro F1 감소폭을 비교하여 강건성을 평가합니다.

<br>

### 2-3. 멀티모달 모델 학습 (추후 업데이트)

### 2-4. 멀티모달 모델 테스트 (추후 업데이트)

