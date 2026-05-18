#!/usr/bin/env bash
# regression_test_trainer.sh
#
# Compares trainer output between origin/modelica_export_copilot and the current
# branch by running training on two datasets (SHF and SR) for both versions.
# All runs are logged to a shared MLflow tracking directory so they can be
# compared side-by-side in the MLflow web UI.
#
# Usage (from any directory):
#   bash bnode/bnode-core/tests/regression_test_trainer.sh
#
# The script must be run with the superproject .venv already activated, or it
# will activate it automatically if it is found at the superproject root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BNODE_CORE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SUPERPROJECT_DIR="$(cd "$BNODE_CORE_DIR/../.." && pwd)"
MLRUNS_DIR="$SCRIPT_DIR/regression_mlruns"

SR_DATASET="resources/data/surrogate-test-data/data/datasets/SimpleSeriesResonance_v4_c-RROCS__n-100_pytest/SimpleSeriesResonance_v4_c-RROCS__n-100_pytest_dataset.hdf5"
SHF_DATASET="resources/data/surrogate-test-data/data/datasets/StratifiedHeatFlowModel_v3_c-RROCS__n-100_pytest/StratifiedHeatFlowModel_v3_c-RROCS__n-100_pytest_dataset.hdf5"

# ---------------------------------------------------------------------------
# Activate venv if not already active
# ---------------------------------------------------------------------------
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    VENV="$SUPERPROJECT_DIR/.venv/bin/activate"
    if [[ ! -f "$VENV" ]]; then
        echo "ERROR: Could not find .venv at $SUPERPROJECT_DIR/.venv" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$VENV"
fi

# ---------------------------------------------------------------------------
# Save current HEAD and set up restore trap
# ---------------------------------------------------------------------------
ORIGINAL_HEAD=$(git -C "$BNODE_CORE_DIR" rev-parse HEAD)
ORIGINAL_BRANCH=$(git -C "$BNODE_CORE_DIR" symbolic-ref --short HEAD 2>/dev/null || echo "")

restore_git_state() {
    echo ""
    echo "=== Restoring bnode-core to original state ==="
    if [[ -n "$ORIGINAL_BRANCH" ]]; then
        git -C "$BNODE_CORE_DIR" checkout "$ORIGINAL_BRANCH" --quiet
    else
        git -C "$BNODE_CORE_DIR" checkout "$ORIGINAL_HEAD" --quiet
    fi
}
trap restore_git_state EXIT

# ---------------------------------------------------------------------------
# Helper: run training from bnode-core root
# ---------------------------------------------------------------------------
run_training() {
    local run_name="$1"
    local dataset_path="$2"
    local label="$3"

    echo ""
    echo "=== [$label] run=$run_name dataset=$(basename "$(dirname "$dataset_path")")"
    (
        cd "$BNODE_CORE_DIR"
        python -m bnode_core.ode.trainer \
            "dataset_path=$dataset_path" \
            "mlflow_experiment_name=regression" \
            "mlflow_run_name=$run_name" \
            "mlflow_tracking_uri=file://$MLRUNS_DIR" \
            "nn_model.training.max_epochs_override=2000" \
            "nn_model.training.early_stopping_patience_override=2000" \
    )
}

# ---------------------------------------------------------------------------
# Check for uncommitted changes that would block the checkout
# ---------------------------------------------------------------------------
if ! git -C "$BNODE_CORE_DIR" diff --quiet HEAD; then
    echo "ERROR: bnode-core has uncommitted changes. Please stash or commit them first." >&2
    exit 1
fi

echo "======================================================="
echo "  Trainer regression test"
echo "======================================================="
echo "  bnode-core:  $BNODE_CORE_DIR"
echo "  Current HEAD: $ORIGINAL_HEAD"
echo "  MLflow dir:  $MLRUNS_DIR"
echo "======================================================="

# ---------------------------------------------------------------------------
# OLD VERSION: origin/modelica_export_copilot
# ---------------------------------------------------------------------------
echo ""
echo "=== Switching bnode-core to origin/modelica_export_copilot ==="
git -C "$BNODE_CORE_DIR" checkout origin/modelica_export_copilot --quiet

run_training "regression_old" "$SHF_DATASET" "1/4 old+SHF"
run_training "regression_old" "$SR_DATASET"  "2/4 old+SR"

# ---------------------------------------------------------------------------
# NEW VERSION: current branch
# ---------------------------------------------------------------------------
echo ""
echo "=== Restoring bnode-core to current branch ==="
trap - EXIT  # disable trap; we restore manually here
restore_git_state

run_training "regression_new" "$SHF_DATASET" "3/4 new+SHF"
run_training "regression_new" "$SR_DATASET"  "4/4 new+SR"

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------
echo ""
echo "======================================================="
echo "  All training runs complete."
echo "======================================================="
echo ""
echo "To compare runs in the MLflow web UI:"
echo ""
echo "  mlflow ui --backend-store-uri \"file://$MLRUNS_DIR\" --port 5000"
echo ""
echo "Then open: http://localhost:5000"
echo ""
echo "  Experiment 'regression_old'  →  origin/modelica_export_copilot"
echo "  Experiment 'regression_new'  →  current branch ($ORIGINAL_BRANCH)"
echo ""
echo "Select runs from both experiments and use the 'Compare' button to"
echo "overlay metrics (loss, validation, test) across both versions."
echo ""
echo "MLflow artifacts are stored in:"
echo "  $MLRUNS_DIR"
