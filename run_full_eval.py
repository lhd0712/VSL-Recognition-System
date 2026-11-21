import json
import time
import numpy as np
import pandas as pd
import keras
from keras import layers, ops
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
test_df = pd.read_csv('data/processed/splits/test_front.csv')
X_test = np.load('data/processed/cache/X_test_front_450D.npy')
y_test = np.load('data/processed/cache/y_test_front.npy')

print(f"Loaded test data: X={X_test.shape}, y={y_test.shape}, df={len(test_df)}")

# 2. Register layer and load best model
@keras.saving.register_keras_serializable(package="VSLAttention")
class TemporalMLPAttention(layers.Layer):
    def __init__(self, units: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.W = self.add_weight(
            name="attention_weight",
            shape=(hidden_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="attention_context_vector",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        u_t = ops.tanh(ops.matmul(inputs, self.W) + self.b)
        score = ops.matmul(u_t, self.u)
        alpha = ops.softmax(score, axis=1)
        context = ops.sum(inputs * alpha, axis=1)
        return context

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

model_path = 'experiments/checkpoints/vsl400_exp2_cosine_best.keras'
model = keras.models.load_model(model_path, compile=False)
print(f"Loaded model successfully from {model_path}")

# 3. Model Inference & Prediction Probabilities
preds = model.predict(X_test, batch_size=128, verbose=0)
pred_labels = np.argmax(preds, axis=1)

# Top-5 predictions
top5_preds = np.argsort(preds, axis=1)[:, -5:]
top5_correct = np.array([y in top5_preds[i] for i, y in enumerate(y_test)])
top1_correct = (pred_labels == y_test)

overall_top1 = np.mean(top1_correct) * 100
overall_top5 = np.mean(top5_correct) * 100
print(f"Overall Test: Top-1 = {overall_top1:.2f}%, Top-5 = {overall_top5:.2f}%")

# 4. Signer Breakdown
test_df['signer_id_str'] = test_df['signer_id'].astype(str).str.zfill(3)
signers = sorted(test_df['signer_id_str'].unique())

signer_metrics = {}
print("\n" + "=" * 60)
print(f"{'Signer':<10}{'Part':<10}{'Samples':<10}{'Top-1 (%)':<15}{'Top-5 (%)':<15}")
print("=" * 60)

for s in signers:
    idx = test_df[test_df['signer_id_str'] == s].index.values
    s_top1 = np.mean(top1_correct[idx]) * 100
    s_top5 = np.mean(top5_correct[idx]) * 100
    part = test_df.loc[idx, 'part'].iloc[0]
    signer_metrics[s] = {
        'signer_id': s,
        'part': int(part),
        'samples': int(len(idx)),
        'top_1': float(round(s_top1, 2)),
        'top_5': float(round(s_top5, 2))
    }
    print(f"{s:<10}{part:<10}{len(idx):<10}{s_top1:>8.2f}%{'':<6}{s_top5:>8.2f}%")

# 5. Length Bracket Breakdown
test_df['length_bracket'] = pd.cut(
    test_df['num_frames'],
    bins=[0, 45, 75, 500],
    labels=['Ngắn (<45)', 'Trung bình (45-75)', 'Dài (>75)']
)
bracket_metrics = {}
print("\n" + "=" * 60)
print(f"{'Thời lượng':<20}{'Samples':<10}{'Top-1 (%)':<15}{'Top-5 (%)':<15}")
print("=" * 60)
for b in ['Ngắn (<45)', 'Trung bình (45-75)', 'Dài (>75)']:
    idx = test_df[test_df['length_bracket'] == b].index.values
    b_top1 = np.mean(top1_correct[idx]) * 100
    b_top5 = np.mean(top5_correct[idx]) * 100
    bracket_metrics[b] = {
        'samples': int(len(idx)),
        'top_1': float(round(b_top1, 2)),
        'top_5': float(round(b_top5, 2))
    }
    print(f"{b:<20}{len(idx):<10}{b_top1:>8.2f}%{'':<6}{b_top5:>8.2f}%")

# 6. Latency Benchmark (Single Sample Streaming Simulation)
print("\n" + "=" * 60)
print("BENCHMARK ĐỘ TRỄ SUY LUẬN (SINGLE-SAMPLE INFERENCE LATENCY)")
print("=" * 60)

# Warmup 50 runs
dummy_sample = X_test[:1]
for _ in range(50):
    _ = model(dummy_sample, training=False)

latencies = []
num_benchmark_runs = 500
for i in range(num_benchmark_runs):
    sample = X_test[i:i+1]
    t0 = time.perf_counter()
    _ = model(sample, training=False)
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000.0) # ms

latencies = np.array(latencies)
mean_latency = float(np.mean(latencies))
median_latency = float(np.median(latencies))
p95_latency = float(np.percentile(latencies, 95))
p99_latency = float(np.percentile(latencies, 99))
throughput_fps = float(1000.0 / mean_latency)

print(f"Mean Latency:   {mean_latency:.2f} ms")
print(f"Median Latency: {median_latency:.2f} ms")
print(f"P95 Latency:    {p95_latency:.2f} ms")
print(f"P99 Latency:    {p99_latency:.2f} ms")
print(f"Throughput FPS: {throughput_fps:.1f} FPS (Single-stream)")

# 7. Class Accuracy Distribution
class_accs = []
for c in range(400):
    c_idx = np.where(y_test == c)[0]
    if len(c_idx) > 0:
        c_acc = np.mean(pred_labels[c_idx] == c) * 100
        class_accs.append(c_acc)

class_accs = np.array(class_accs)
acc_100 = int(np.sum(class_accs == 100.0))
acc_ge_80 = int(np.sum(class_accs >= 80.0))
acc_ge_50 = int(np.sum(class_accs >= 50.0))
acc_lt_50 = int(np.sum(class_accs < 50.0))

print("\n" + "=" * 60)
print("PHÂN BỐ ĐỘ CHÍNH XÁC TRÊN 400 LỚP TỪ VỰNG:")
print(f"  - Đạt tuyệt đối 100% Top-1 : {acc_100} từ ({acc_100/400*100:.1f}%)")
print(f"  - Đạt >= 80% Top-1          : {acc_ge_80} từ ({acc_ge_80/400*100:.1f}%)")
print(f"  - Đạt >= 50% Top-1          : {acc_ge_50} từ ({acc_ge_50/400*100:.1f}%)")
print(f"  - Dưới 50% Top-1            : {acc_lt_50} từ ({acc_lt_50/400*100:.1f}%)")
