.PHONY: run-main-multiseed run-capacity-sweep run-ablations run-long-rollouts run-resolution-transfer analyze plots all-paper-results

run-main-multiseed:
	python run_all_experiments.py --models current_state lag_stack history2history temporal_unet convlstm temporal_transformer hs_fno

run-capacity-sweep:
	python scripts/run_capacity_sweep.py

run-ablations:
	python run_all_experiments.py --models hs_fno hs_fno_no_shift hs_fno_no_delay_conditioning hs_fno_coord_conditioning hs_fno_film_conditioning hs_fno_rollout_semiflow hs_transformer hsno_unet

run-long-rollouts:
	python run_all_experiments.py --models current_state lag_stack history2history temporal_unet convlstm temporal_transformer hs_fno

run-resolution-transfer:
	python run_all_experiments.py --models hs_fno current_state lag_stack history2history

analyze:
	python analyze_all_results.py
	python scripts/check_claims.py
	python scripts/audit_diagnostics.py
	python scripts/summarize_efficiency.py
	python scripts/summarize_solver_comparison.py

plots:
	python scripts/generate_result_plots.py

all-paper-results: run-main-multiseed run-capacity-sweep run-ablations analyze plots
