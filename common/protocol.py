import json
import numpy as np


class RunLedger:
    """공통 실행 원장 (RunLedger JSONL)"""
    def __init__(self, filepath="run_ledger.jsonl"):
        self.filepath = filepath

    def log(self, entry):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def paired_cluster_bootstrap(data_a, data_b, num_samples=1000, alpha=0.05):
    """Paired Cluster Bootstrap implementation."""
    diffs = np.array(data_a) - np.array(data_b)
    n = len(diffs)
    boot_means = []
    for _ in range(num_samples):
        sample = np.random.choice(diffs, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(boot_means)), (float(lower), float(upper))


def compute_primal_dual_gap_integral(gaps, time_steps):
    """P4용 Primal-Dual Gap Integral 계산 함수."""
    return float(np.trapz(gaps, time_steps))
