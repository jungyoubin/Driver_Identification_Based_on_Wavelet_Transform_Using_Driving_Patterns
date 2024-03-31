import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# 데이터 로드
df = pd.read_csv('./11_CAN_window60.csv', encoding='cp949')
df = df.sample(frac=1, random_state=7777).reset_index(drop=True)

use_col = ['RearHeatPower266', 'DI_accelPedalPos', 'SmoothBattCurrent132', 'VCFRONT_chillerExvFlowm3',
           'BMS_maxDischargePower', 'DI_regenLight', 'DI_regenLight', 'DIR_torqueActual',
           'DIR_torqueCommand', 'DIR_torqueActual', 'SystemHeatPowerMax268', 'SystemHeatPower268',
           'VCFRONT_pumpBatteryRPMActualm0', 'BattVoltage132', 'SteeringSpeed129', 'SteeringAngle129',
           'DI_vehicleSpeed', 'label', 'drive_count', 'course']

# feature 몇 개 제외
df = df[[col for col in df.columns if any(substring in col for substring in use_col)]]

df_train = df[df['drive_count'] != 0]
df_valid = df[df['drive_count'] == 0]

train_data = df_train.drop(['course', 'drive_count'], axis=1)
valid_data = df_valid.drop(['course', 'drive_count'], axis=1)

train_data.reset_index(inplace=True, drop=True)
valid_data.reset_index(inplace=True, drop=True)

X_train = train_data.drop('label', axis=1)
y_train = train_data['label']

X_valid = valid_data.drop('label', axis=1)
y_valid = valid_data['label']

# 결측치 drop
X_train = X_train.dropna()
y_train = y_train[X_train.index]

X_valid = X_valid.dropna()
y_valid = y_valid[X_valid.index]

# XGBoost 모델 GPU로 학습
import xgboost as xgb

xgb_model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=1000, random_state=42, tree_method='gpu_hist')  # GPU 사용 설정
xgb_model.fit(X_train, y_train)

from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.preprocessing import label_binarize

# 예측 확률 계산
y_pred_prob = xgb_model.predict_proba(X_valid)

# 레이블에 따라 확률을 그룹화
grouped_prob = {label: [] for label in np.unique(y_train)}
for prob, label in zip(y_pred_prob, y_valid):
    grouped_prob[label].append(prob)


# TOP-3 정확도 계산 함수
def top_3_accuracy(probs, labels):
    correct = 0
    for prob, label in zip(probs, labels):
        top_3_labels = np.argsort(prob)[-3:]
        correct += label in top_3_labels
    return correct / len(labels)


# 다중 클래스 MAP 계산 함수
def mean_average_precision_multiclass(probs, y_true, n_classes):
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    average_precisions = []
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
        ap = np.trapz(recall, precision)
        average_precisions.append(ap)
    return np.mean(average_precisions)


# F1 스코어 계산 함수
def f1_score(y_true, y_pred):
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return f1


# 예측을 집계하고 평가 지표를 계산하는 함수
def evaluate_performance(grouped_prob, vote_count, n_classes):
    y_true = []
    y_pred = []
    y_scores = []
    for label, probs in grouped_prob.items():
        for i in range(0, len(probs), vote_count):
            avg_prob = np.mean(probs[i:i + vote_count], axis=0)
            predicted_label = np.argmax(avg_prob)
            y_true.append(label)
            y_pred.append(predicted_label)
            y_scores.append(avg_prob)

    acc = np.mean(np.array(y_pred) == np.array(y_true))
    top_3_acc = top_3_accuracy(np.array(y_scores), np.array(y_true))
    map_score = mean_average_precision_multiclass(np.array(y_scores), np.array(y_true), n_classes)
    f1 = f1_score(np.array(y_true), np.array(y_pred))

    return acc, top_3_acc, map_score, f1


# all voting
def evaluate_performance_all(grouped_prob):
    y_true = []
    y_pred = []
    y_scores = []
    for idx in range(len(grouped_prob)):
        # 각 드라이버의 예측 확률값
        driver = grouped_prob[idx]
        # 클래스별 평균 확률값 계산
        avg_prob = np.mean(np.array(driver), axis=0)
        # 가장 높은 확률을 가진 클래스 선택
        predicted_label = np.argmax(avg_prob)

        y_true.append(idx)
        y_pred.append(predicted_label)
        y_scores.append(avg_prob)

    acc = np.mean(np.array(y_pred) == np.array(y_true))
    top_3_acc = top_3_accuracy(np.array(y_scores), np.array(y_true))
    # print(y_scores)
    # print(y_true)
    map_score = mean_average_precision_multiclass(np.array(y_scores), np.array(y_true), 11)
    f1 = f1_score(np.array(y_true), np.array(y_pred))

    return acc, top_3_acc, map_score, f1


# 각 보팅 방식에 대한 평가 지표 계산
n_classes = len(np.unique(y_train))
for vote_count in [1, 3, 10]:
    acc, top_3_acc, map_score, f1 = evaluate_performance(grouped_prob, vote_count, n_classes)
    print(
        f"Voting {vote_count}: 정확도 = {np.round(acc * 100, 1)}, TOP-3 정확도 = {np.round(top_3_acc * 100, 1)}, MAP = {np.round(map_score * 100, 1)}, F1 스코어 = {np.round(f1 * 100, 1)}")

# all voting
acc, top_3_acc, map_score, f1 = evaluate_performance_all(grouped_prob)
print(
    f"Voting all: 정확도 = {np.round(acc * 100, 1)}, TOP-3 정확도 = {np.round(top_3_acc * 100, 1)}, MAP = {np.round(map_score * 100, 1)}, F1 스코어 = {np.round(f1 * 100, 1)}")