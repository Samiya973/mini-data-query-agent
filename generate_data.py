import random
import csv

random.seed(42)
rows = []
run_id = 1
for i in range(24):
    lr = round(random.choice([0.01, 0.02, 0.03, 0.05, 0.08, 0.1]), 3)
    num_leaves = random.choice([15, 31, 63, 127])
    max_depth = random.choice([-1, 4, 6, 8, 10])
    n_estimators = random.choice([100, 200, 300, 500, 800])
    min_child_samples = random.choice([10, 20, 30, 50])
    base = 0.78
    base += 0.05 if num_leaves in (63, 127) else 0.0
    base += 0.03 if lr in (0.03, 0.05) else -0.01
    base += 0.02 if n_estimators >= 300 else 0.0
    noise = random.uniform(-0.02, 0.02)
    roc_auc = round(min(0.97, max(0.70, base + noise)), 4)
    precision = round(roc_auc - random.uniform(0.05, 0.12), 4)
    recall = round(roc_auc - random.uniform(0.08, 0.15), 4)
    f1 = round(2 * precision * recall / (precision + recall), 4)
    top_shap_feature = random.choice([
        'debt_to_income_ratio', 'credit_utilization', 'payment_history_score',
        'num_delinquent_accounts', 'annual_income', 'credit_age_months'
    ])
    date = f'2026-{random.randint(1,6):02d}-{random.randint(1,28):02d}'
    rows.append([run_id, date, lr, num_leaves, max_depth, n_estimators,
                 min_child_samples, roc_auc, precision, recall, f1, top_shap_feature])
    run_id += 1

header = ['run_id', 'date', 'learning_rate', 'num_leaves', 'max_depth',
          'n_estimators', 'min_child_samples', 'roc_auc', 'precision',
          'recall', 'f1_score', 'top_shap_feature']

with open('data/experiment_runs.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to data/experiment_runs.csv")