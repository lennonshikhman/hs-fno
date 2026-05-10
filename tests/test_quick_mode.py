import os
import subprocess
import sys

import pandas as pd


def test_quick_mode_entrypoint_invokes(tmp_path):
    env = os.environ.copy()
    env["HSNO_QUICK_TEST_MINIMAL"] = "1"
    cmd = [sys.executable, "run_all_experiments.py", "--quick", "--overwrite", "--output-dir", str(tmp_path / "outputs")]
    result = subprocess.run(cmd, env=env, cwd=os.getcwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    metrics_path = tmp_path / "outputs" / "metrics" / "all_metrics.csv"
    assert metrics_path.exists()
    metrics = pd.read_csv(metrics_path)
    assert "seed" in metrics.columns
    assert metrics["seed"].nunique() == 1
    assert (tmp_path / "outputs" / "metrics" / "summary_by_model.csv").exists()
