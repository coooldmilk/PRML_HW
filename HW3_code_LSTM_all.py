# ============================================================
# PM2.5 Forecasting Code
# Methods:
# 1. Persistence Baseline
# 2. Standard LSTM
# 3. LSTM + Temporal Attention
# 4. Residual LSTM
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# 1. 参数配置
# ============================================================

MAIN_DATA_PATH = r"D:\啦啦啦啦啦啦啦我的东西\大学\大三下2026春\模式识别与机器学习\作业\HW3\archive\LSTM-Multivariate_pollution.csv"
FINAL_TEST_PATH = r"D:\啦啦啦啦啦啦啦我的东西\大学\大三下2026春\模式识别与机器学习\作业\HW3\archive\pollution_test_data1.csv"

LOOKBACK = 12
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2
SEED = 42

CLIP_NEGATIVE_PREDICTIONS = True
PLOT_N = 500

TARGET_COL = "pollution"
NUMERIC_FEATURE_COLS = [
    "pollution", "dew", "temp", "press", "wnd_spd", "snow", "rain"
]
CATEGORICAL_FEATURE_COL = "wnd_dir"

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. 工具函数
# ============================================================

def get_onehot_encoder():
    """OneHotEncoder"""
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def get_ohe_columns(encoder, col_name):
    """one-hot 列名获取"""
    try:
        return encoder.get_feature_names_out([col_name])
    except AttributeError:
        return encoder.get_feature_names([col_name])


def preprocess_missing_values(dataframe):
    """处理缺失值：数值变量插值，类别变量前后填充。"""
    df_processed = dataframe.copy()

    if df_processed.isna().sum().sum() > 0:
        existing_numeric_cols = [
            col for col in NUMERIC_FEATURE_COLS if col in df_processed.columns
        ]
        df_processed[existing_numeric_cols] = (
            df_processed[existing_numeric_cols]
            .interpolate(method="linear")
            .ffill()
            .bfill()
        )

        if CATEGORICAL_FEATURE_COL in df_processed.columns:
            df_processed[CATEGORICAL_FEATURE_COL] = (
                df_processed[CATEGORICAL_FEATURE_COL].ffill().bfill()
            )

    return df_processed


def create_sequences(X_scaled, y_abs, dates=None, lookback=12):
    """
    构造监督学习样本

    X_t: 过去 lookback 小时的多变量序列
    y_t: 当前小时 pollution
    prev_t: 上一小时 pollution
    delta_t: 当前小时与上一小时 pollution 的变化量
    """
    X_seq, y_abs_seq, prev_abs_seq, y_delta_seq, target_dates = [], [], [], [], []

    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback:i, :])

        current_pollution = y_abs[i]
        previous_pollution = y_abs[i - 1]
        delta_pollution = current_pollution - previous_pollution

        y_abs_seq.append(current_pollution)
        prev_abs_seq.append(previous_pollution)
        y_delta_seq.append(delta_pollution)

        if dates is not None:
            target_dates.append(dates.iloc[i])

    if dates is not None:
        target_dates = pd.to_datetime(target_dates)
    else:
        target_dates = None

    return (
        np.array(X_seq),
        np.array(y_abs_seq),
        np.array(prev_abs_seq),
        np.array(y_delta_seq),
        target_dates,
    )


def clip_predictions(y_pred):
    """PM2.5 负预测值裁剪为0"""
    if CLIP_NEGATIVE_PREDICTIONS:
        return np.maximum(y_pred, 0)
    return y_pred

#评价指标
def regression_metrics(y_true, y_pred):
    """计算 MAE、RMSE、R2 和 sMAPE。"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    denominator = np.abs(y_true) + np.abs(y_pred)
    smape = np.mean(
        np.where(denominator == 0, 0, 2 * np.abs(y_pred - y_true) / denominator)
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "sMAPE(%)": smape,
    }


def evaluate_persistence(y_abs_true, prev_abs, model_name="Persistence Baseline"):
    """Persistence Baseline: y_hat(t) = y(t-1)。"""
    y_pred = clip_predictions(prev_abs.copy())
    metrics = regression_metrics(y_abs_true, y_pred)
    metrics["Model"] = model_name
    return metrics, y_pred


def evaluate_absolute_model(model, X, y_abs_true, abs_target_scaler, model_name):
    """评估直接预测 pollution 绝对值的模型。"""
    y_pred_scaled = model.predict(X, verbose=0).reshape(-1)
    y_pred = abs_target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).reshape(-1)
    y_pred = clip_predictions(y_pred)

    metrics = regression_metrics(y_abs_true, y_pred)
    metrics["Model"] = model_name
    return metrics, y_pred


def evaluate_residual_model(model, X, y_abs_true, prev_abs, delta_scaler, model_name):
    """评估 Residual LSTM: y_hat(t) = y(t-1) + delta_hat(t)。"""
    delta_pred_scaled = model.predict(X, verbose=0).reshape(-1)
    delta_pred = delta_scaler.inverse_transform(
        delta_pred_scaled.reshape(-1, 1)
    ).reshape(-1)

    y_pred = prev_abs + delta_pred
    y_pred = clip_predictions(y_pred)

    metrics = regression_metrics(y_abs_true, y_pred)
    metrics["Model"] = model_name
    return metrics, y_pred


def print_metrics_table(title, metrics_list):
    """输出最终评价指标。"""
    results_df = pd.DataFrame(metrics_list)
    results_df = results_df[["Model", "MAE", "RMSE", "R2", "sMAPE(%)"]]

    print(f"\n{title}")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return results_df


# ============================================================
# 3. 模型定义
# ============================================================

class TemporalAttention(Layer):
    """Temporal Attention Layer over LSTM hidden states."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.W = self.add_weight(
            name="attention_weight_matrix",
            shape=(hidden_dim, hidden_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(hidden_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.v = self.add_weight(
            name="attention_score_vector",
            shape=(hidden_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        score_first_part = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        scores = tf.tensordot(score_first_part, self.v, axes=1)
        scores = tf.squeeze(scores, axis=-1)
        attention_weights = tf.nn.softmax(scores, axis=1)
        context_vector = tf.reduce_sum(
            inputs * tf.expand_dims(attention_weights, axis=-1), axis=1
        )
        return context_vector, attention_weights


def compile_model(model):
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_standard_lstm_model(lookback, n_features):
    """LSTM: 直接预测 pollution(t)"""
    inputs = Input(shape=(lookback, n_features))
    x = LSTM(64, return_sequences=False)(inputs)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)
    return compile_model(Model(inputs=inputs, outputs=outputs, name="Standard_LSTM"))


def build_attention_lstm_model(lookback, n_features):
    """
    LSTM + Temporal Attention
    context vector 与最后一个 hidden state 拼接后预测 pollution(t)
    """
    inputs = Input(shape=(lookback, n_features))
    lstm_outputs, last_hidden_state, _ = LSTM(
        64, return_sequences=True, return_state=True
    )(inputs)
    lstm_outputs = Dropout(DROPOUT_RATE)(lstm_outputs)

    context_vector, _attention_weights = TemporalAttention(name="temporal_attention")(
        lstm_outputs
    )
    combined = Concatenate()([context_vector, last_hidden_state])

    x = Dense(32, activation="relu")(combined)
    x = Dropout(DROPOUT_RATE)(x)
    outputs = Dense(1)(x)
    return compile_model(
        Model(inputs=inputs, outputs=outputs, name="LSTM_with_Temporal_Attention")
    )


def build_residual_lstm_model(lookback, n_features):
    """Residual LSTM: 
    预测 delta(t) = pollution(t) - pollution(t-1)"""
    inputs = Input(shape=(lookback, n_features))
    x = LSTM(64, return_sequences=False)(inputs)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)
    outputs = Dense(1)(x)
    return compile_model(Model(inputs=inputs, outputs=outputs, name="Residual_LSTM"))


def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]


# ============================================================
# 4. 数据
# ============================================================

# 主数据集
df = pd.read_csv(MAIN_DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df = preprocess_missing_values(df)
df["year"] = df["date"].dt.year

train_row_mask = df["year"].between(2010, 2012)
val_row_mask = df["year"] == 2013
test_row_mask = df["year"] == 2014

# One-hot encoding for wind direction, fitted only on training data.
ohe = get_onehot_encoder()
ohe.fit(df.loc[train_row_mask, [CATEGORICAL_FEATURE_COL]])
ohe_cols = get_ohe_columns(ohe, CATEGORICAL_FEATURE_COL)

wnd_encoded_df = pd.DataFrame(
    ohe.transform(df[[CATEGORICAL_FEATURE_COL]]),
    columns=ohe_cols,
    index=df.index,
)

numeric_feature_df = df[NUMERIC_FEATURE_COLS].astype(float)
feature_df = pd.concat([numeric_feature_df, wnd_encoded_df], axis=1)
feature_cols = feature_df.columns.tolist()
target_abs = df[TARGET_COL].astype(float).values

# Correlation matrix prepared here; plotted at the end.
correlation_df = df[NUMERIC_FEATURE_COLS].astype(float).corr()

# Feature scaler fitted only on training data.
feature_scaler = MinMaxScaler()
feature_scaler.fit(feature_df.loc[train_row_mask])
X_all_scaled = feature_scaler.transform(feature_df)

X_seq, y_abs_seq, prev_abs_seq, y_delta_seq, target_dates = create_sequences(
    X_scaled=X_all_scaled,
    y_abs=target_abs,
    dates=df["date"],
    lookback=LOOKBACK,
)

target_years = target_dates.year
train_seq_mask = (target_years >= 2010) & (target_years <= 2012)
val_seq_mask = target_years == 2013
test_seq_mask = target_years == 2014

X_train, X_val, X_test = X_seq[train_seq_mask], X_seq[val_seq_mask], X_seq[test_seq_mask]
y_abs_train, y_abs_val, y_abs_test = (
    y_abs_seq[train_seq_mask],
    y_abs_seq[val_seq_mask],
    y_abs_seq[test_seq_mask],
)
prev_abs_train, prev_abs_val, prev_abs_test = (
    prev_abs_seq[train_seq_mask],
    prev_abs_seq[val_seq_mask],
    prev_abs_seq[test_seq_mask],
)
y_delta_train, y_delta_val, y_delta_test = (
    y_delta_seq[train_seq_mask],
    y_delta_seq[val_seq_mask],
    y_delta_seq[test_seq_mask],
)
test_dates = target_dates[test_seq_mask]
n_features = X_train.shape[2]

# Target scaling.
abs_target_scaler = MinMaxScaler()
abs_target_scaler.fit(y_abs_train.reshape(-1, 1))
y_abs_train_scaled = abs_target_scaler.transform(y_abs_train.reshape(-1, 1)).reshape(-1)
y_abs_val_scaled = abs_target_scaler.transform(y_abs_val.reshape(-1, 1)).reshape(-1)

delta_scaler = StandardScaler()
delta_scaler.fit(y_delta_train.reshape(-1, 1))
y_delta_train_scaled = delta_scaler.transform(y_delta_train.reshape(-1, 1)).reshape(-1)
y_delta_val_scaled = delta_scaler.transform(y_delta_val.reshape(-1, 1)).reshape(-1)


# ============================================================
# 5. 模型训练
# ============================================================

standard_lstm_model = build_standard_lstm_model(LOOKBACK, n_features)
attention_lstm_model = build_attention_lstm_model(LOOKBACK, n_features)
residual_lstm_model = build_residual_lstm_model(LOOKBACK, n_features)

history_standard = standard_lstm_model.fit(
    X_train,
    y_abs_train_scaled,
    validation_data=(X_val, y_abs_val_scaled),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    callbacks=get_callbacks(),
    verbose=0,
)

history_attention = attention_lstm_model.fit(
    X_train,
    y_abs_train_scaled,
    validation_data=(X_val, y_abs_val_scaled),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    callbacks=get_callbacks(),
    verbose=0,
)

history_residual = residual_lstm_model.fit(
    X_train,
    y_delta_train_scaled,
    validation_data=(X_val, y_delta_val_scaled),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    callbacks=get_callbacks(),
    verbose=0,
)


# ============================================================
# 6. 2014 Test Set
# ============================================================

persistence_metrics, y_pred_persistence = evaluate_persistence(
    y_abs_test, prev_abs_test, "Persistence Baseline"
)
standard_metrics, y_pred_standard = evaluate_absolute_model(
    standard_lstm_model, X_test, y_abs_test, abs_target_scaler, "Standard LSTM"
)
attention_metrics, y_pred_attention = evaluate_absolute_model(
    attention_lstm_model,
    X_test,
    y_abs_test,
    abs_target_scaler,
    "LSTM + Temporal Attention",
)
residual_metrics, y_pred_residual = evaluate_residual_model(
    residual_lstm_model, X_test, y_abs_test, prev_abs_test, delta_scaler, "Residual LSTM"
)

results_df = print_metrics_table(
    "2014 Test Set Performance",
    [persistence_metrics, standard_metrics, attention_metrics, residual_metrics],
)


# ============================================================
# 7. Optional Evaluation: External Test Data
# ============================================================

def prepare_external_data(external_path, onehot_encoder, feature_scaler, feature_cols, lookback=12):
    external_df = pd.read_csv(external_path)
    external_df = preprocess_missing_values(external_df)

    external_wnd_encoded_df = pd.DataFrame(
        onehot_encoder.transform(external_df[[CATEGORICAL_FEATURE_COL]]),
        columns=get_ohe_columns(onehot_encoder, CATEGORICAL_FEATURE_COL),
        index=external_df.index,
    )

    external_numeric_feature_df = external_df[NUMERIC_FEATURE_COLS].astype(float)
    external_feature_df = pd.concat(
        [external_numeric_feature_df, external_wnd_encoded_df], axis=1
    )
    external_feature_df = external_feature_df.reindex(columns=feature_cols, fill_value=0)

    external_X_scaled = feature_scaler.transform(external_feature_df)
    external_y_abs = external_df[TARGET_COL].astype(float).values

    return create_sequences(
        X_scaled=external_X_scaled,
        y_abs=external_y_abs,
        dates=None,
        lookback=lookback,
    )[:4]


external_results_df = None
external_predictions = None

if os.path.exists(FINAL_TEST_PATH):
    X_external, y_external_abs, prev_external_abs, y_external_delta = prepare_external_data(
        FINAL_TEST_PATH, ohe, feature_scaler, feature_cols, LOOKBACK
    )

    external_persistence_metrics, external_pred_persistence = evaluate_persistence(
        y_external_abs, prev_external_abs, "Persistence Baseline"
    )
    external_standard_metrics, external_pred_standard = evaluate_absolute_model(
        standard_lstm_model,
        X_external,
        y_external_abs,
        abs_target_scaler,
        "Standard LSTM",
    )
    external_attention_metrics, external_pred_attention = evaluate_absolute_model(
        attention_lstm_model,
        X_external,
        y_external_abs,
        abs_target_scaler,
        "LSTM + Temporal Attention",
    )
    external_residual_metrics, external_pred_residual = evaluate_residual_model(
        residual_lstm_model,
        X_external,
        y_external_abs,
        prev_external_abs,
        delta_scaler,
        "Residual LSTM",
    )

    external_results_df = print_metrics_table(
        "External Test Data Performance",
        [
            external_persistence_metrics,
            external_standard_metrics,
            external_attention_metrics,
            external_residual_metrics,
        ],
    )

    external_predictions = {
        "Actual PM2.5": y_external_abs,
        "Persistence Baseline": external_pred_persistence,
        "Standard LSTM": external_pred_standard,
        "LSTM + Temporal Attention": external_pred_attention,
        "Residual LSTM": external_pred_residual,
    }


# ============================================================
# 画图
# ============================================================

COLORS = {
    "Actual PM2.5": "gray",
    "Persistence Baseline": "tab:orange",
    "LSTM": "tab:blue",
    "LSTM + Temporal Attention": "tab:red",
    "Residual LSTM": "tab:green",
}

model_predictions = {
    "Persistence Baseline": y_pred_persistence,
    "LSTM": y_pred_standard,
    "LSTM + Temporal Attention": y_pred_attention,
    "Residual LSTM": y_pred_residual,
}


for model_name, y_pred in model_predictions.items():
    plt.figure(figsize=(14, 5))
    plt.plot(
        test_dates[:PLOT_N],
        y_abs_test[:PLOT_N],
        label="Actual PM2.5",
        color=COLORS["Actual PM2.5"],
        linewidth=1.8,
    )
    plt.plot(
        test_dates[:PLOT_N],
        y_pred[:PLOT_N],
        label=model_name,
        color=COLORS[model_name],
        linewidth=1.5,
    )
    plt.title(f"Actual vs Predicted PM2.5: {model_name}")
    plt.xlabel("Date")
    plt.ylabel("PM2.5 Pollution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#all
plt.figure(figsize=(15, 6))
plt.plot(
    test_dates[:PLOT_N],
    y_abs_test[:PLOT_N],
    label="Actual PM2.5",
    color=COLORS["Actual PM2.5"],
    linewidth=2.0,
)
for model_name, y_pred in model_predictions.items():
    plt.plot(
        test_dates[:PLOT_N],
        y_pred[:PLOT_N],
        label=model_name,
        color=COLORS[model_name],
        linewidth=1.4,
    )
plt.title("Actual vs Predicted PM2.5 on 2014 Test Set")
plt.xlabel("Date")
plt.ylabel("PM2.5 Pollution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




