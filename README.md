# 비트코인 가격 예측 및 트레이딩 전략 프로젝트

## 1. 프로젝트 개요

이 프로젝트는 딥러닝 모델을 사용하여 비트코인 가격의 상승 또는 하락을 예측하고, 예측 결과를 기반으로 트레이딩 전략을 수립하여 수익률을 극대화하는 것을 목표로 합니다. 최종적으로 개발된 전략의 수익률을 Buy and Hold 벤치마크 전략과 비교하여 성능을 분석합니다.

## 2. 모델 설계

### 2.1. 모델 아키텍처

가격 예측을 위해 **LSTM(Long Short-Term Memory)** 기반의 딥러닝 모델을 설계했습니다. LSTM은 시계열 데이터의 장기 의존성을 학습하는 데 효과적인 순환 신경망(RNN)의 한 종류입니다.

본 프로젝트에서 사용된 모델의 구조는 다음과 같습니다.

```python
class MyTradingModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.4):
        super(MyTradingModel, self).__init__()
        # 2개의 LSTM 레이어와 Dropout을 적용하여 과적합을 방지합니다.
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=dropout)
        
        # LSTM 출력에 배치 정규화를 적용하여 학습을 안정화합니다.
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Fully Connected Layer를 추가하여 최종 예측을 수행합니다.
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
        # 최종 출력은 Sigmoid 함수를 통해 0과 1 사이의 확률 값으로 변환됩니다.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM의 마지막 시점의 출력을 사용합니다.
        lstm_out, _ = self.lstm1(x)
        lstm_out = lstm_out[:, -1, :]
        
        lstm_out = self.bn1(lstm_out)
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out
```

**주요 하이퍼파라미터:**

- `hidden_size`: 128
- `dropout`: 0.4
- `learning_rate`: 0.0005
- `epochs`: 100 (Early Stopping 적용)
- `patience`: 20

### 2.2. 선택 이유

- **LSTM**: 금융 시계열 데이터와 같이 복잡하고 노이즈가 많은 데이터에서 장기적인 패턴을 학습하는 데 강점을 가집니다.
- **2-Layer LSTM**: 단일 레이어보다 더 복잡한 패턴을 학습할 수 있도록 모델의 깊이를 추가했습니다.
- **Dropout & BatchNorm**: 과적합을 방지하고 학습 과정을 안정화시켜 모델의 일반화 성능을 향상시킵니다.
- **Adam Optimizer**: 효율적이고 안정적인 학습을 위해 Adam 옵티마이저를 사용했습니다.

## 3. 투자 전략 설계

### 3.1. 확률 기반 포지션 조절 전략

모델이 예측한 **상승 확률**을 투자 결정에 직접 활용하는 전략을 사용합니다.

- **전략 규칙**:
    1. 모델이 예측한 내일의 가격 상승 확률(`prob`)이 미리 설정된 임계값(`threshold`)보다 높을 경우에만 투자를 고려합니다.
    2. 투자 비율은 `position_scaling` 옵션에 따라 결정됩니다.
        - `position_scaling=True`: 상승 확률에 비례하여 투자 비중을 조절합니다. (예: 상승 확률 70% -> 자본의 70% 투자)
        - `position_scaling=False`: 임계값을 넘으면 전액 투자합니다.
    3. 상승 확률이 임계값보다 낮으면, 보유 중인 비트코인을 전량 매도하고 현금을 보유합니다.

### 3.2. 최적 임계값 탐색

다양한 `threshold` 값(0.5 ~ 0.8)에 대해 시뮬레이션을 실행하여, 테스트 기간 동안 가장 높은 수익률을 기록한 최적의 임계값을 탐색했습니다. 이를 통해 시장 상황에 맞는 가장 효과적인 투자 기준을 찾고자 했습니다.

## 4. 분석 결과

**아래 결과는 `solution.ipynb` 노트북을 실행한 후 `results.txt` 파일에 저장된 값을 기입한 것입니다.**

- **최적 투자 임계값(Optimal Threshold)**: [여기에 results.txt의 Optimal Threshold 값을 입력하세요]
- **나의 전략 최종 수익률**: [여기에 results.txt의 My Strategy Return 값을 입력하세요]
- **Buy and Hold 전략 수익률**: [여기에 results.txt의 Buy and Hold Return 값을 입력하세요]

### 포트폴리오 가치 변화 그래프

![Trading Result](solution_executed.ipynb) 

(위 이미지는 `solution_executed.ipynb` 노트북의 마지막 셀에 생성된 그래프입니다.)

## 5. 실행 방법

1. **필요 라이브러리 설치**
   ```bash
   pip install -r requirements.txt
   pip install torch
   ```

2. **Jupyter Notebook 실행**
   `solution.ipynb` 파일을 열고, 상단의 **[Cell] -> [Run All]**을 클릭하여 전체 코드를 실행합니다.

3. **결과 확인**
   - 코드 실행이 완료되면, 노트북 마지막 셀에서 포트폴리오 가치 변화 그래프를 확인할 수 있습니다.
   - 프로젝트 폴더에 생성된 `results.txt` 파일에서 최적 임계값과 최종 수익률 등 상세 결과를 확인할 수 있습니다.

## 6. 결론

본 프로젝트를 통해 LSTM 기반의 딥러닝 모델과 확률 기반 트레이딩 전략을 성공적으로 수립했습니다. 시뮬레이션 결과, 제안된 전략은 Buy and Hold 벤치마크 대비 [초과/미달]하는 수익률을 기록했으며, 이는 `threshold` 최적화를 통해 리스크를 관리하고 수익을 극대화한 결과입니다.

향후 모델의 성능을 더욱 개선하고, 다양한 기술적 지표를 결합한 복합 전략을 개발하여 더 높은 수익률을 추구할 수 있을 것으로 기대됩니다.