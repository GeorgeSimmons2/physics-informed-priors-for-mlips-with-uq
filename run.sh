#!/bin/bash
# run.sh — reproduce the three results of this repository.
#
#   bash run.sh setup          a fresh environment: .venv/ from requirements.txt, and the Julia environment
#                              (Project.toml + Manifest.toml) in a repo-local depot .julia/   (~40 min, once)
#   bash run.sh data [SOURCE]  bring in the 5 tier-B inputs (DFT data, ACE design matrices) from SOURCE
#                              (default ../physics-informed-priors-repro) and check every SHA-256
#   bash run.sh replot         every figure from the committed published results → repro/replot/  (minutes)
#   bash run.sh saved          re-derive everything from the published fitted models → repro/saved/   (SLURM)
#   bash run.sh refit          refit both ACE models and retrain MACE from the DFT data, then the same
#                              chain on them → repro/refit/                                          (SLURM)
#   bash run.sh all            saved and refit together
#   bash run.sh verify ROUTE   compare repro/ROUTE (saved | refit) with published/ → repro/ROUTE/VERIFY.md
#
# Every route writes only under repro/<route>/ (and model-specific Hessian caches beside each model), so
# published/ is never touched.  A route directory that already exists is refused unless FORCE=1.
# Env: PARTITION (compute), JULIA_DEPOT_PATH (default .julia/), FORCE.  Every job fits one 48-core, ~170 GB node;
# the requests are about twice the peak memory measured for each stage (largest: the hypercube, 43 GB).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO"
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$REPO/.julia}
JL="julia --project=$REPO"
PY=$REPO/.venv/bin/python
PART=${PARTITION:-compute}
LOG=$REPO/repro/logs

need_env()  { [ -x "$PY" ] && [ -d "$JULIA_DEPOT_PATH/compiled" ] || { echo "run first: bash run.sh setup"; exit 1; }; }
need_data() { sha256sum --quiet -c data/tierB.sha256 || { echo "run first: bash run.sh data"; exit 1; }; }

# sub NAME DEPENDENCIES PARTITION TASKS MEM_PER_TASK_MB TIME ENV COMMAND  → job id
sub() {
    sbatch --parsable -p "$3" -N 1 -n "$4" --mem-per-cpu="$5" -t "$6" -J "$1" ${2:+-d afterok:$2} \
        -o "$LOG/$1_%j.log" --export="ALL,$7" --wrap "cd $REPO && $8"
}

# route NAME ACE_DEPENDENCY MACE_DEPENDENCY ACE_MODELS MODEL_FILE — submit the whole chain for one route
route() {
    local N=$1 AD=$2 MD=$3 AM=$4 MF=$5 O=$REPO/repro/$1
    local E="OUT=$O,ACE_MODELS=$AM,MODEL_FILE=$MF"
    local T='-t $SLURM_NTASKS'
    # Al_12: in sequence — 01 builds the 4×4×4 undotted Hessian cache the next three read
    local c1 c2 c3 c3n c4 c5 d1 d2 f4 r1 n1 n2 f2 m1 m2 p1 p2 p3 p4
    c1=$(sub "${N}_al12_01" "$AD" "$PART" 20 3988 06:00:00 "$E" "$JL $T al12/01_committee.jl")
    c2=$(sub "${N}_al12_02" "$c1" "$PART" 42 3988 04:00:00 "$E" "$JL $T al12/02_cutting_plane_cloud.jl")
    c3=$(sub "${N}_al12_03" "$c2" "$PART" 8 12000 04:00:00 "$E" "$JL $T al12/03_hypercube_rejection.jl")
    # verify's yardstick for the committees' test coverage: al12/03 again with 600 members (its first 30 are the
    # committee above), scored as 20 disjoint 30-member committees
    c3n=$(sub "${N}_al12_03_noise" "$c3" "$PART" 8 12000 02:00:00 "$E,OUTDIR=$O/al12/hypercube_noise" \
          "$JL $T al12/03_hypercube_rejection.jl 600 20 && $JL verify/coverage_noise.jl $O/al12/hypercube_noise/parity_calibration_predictions.jls data/Al/manual_df_test_Al.xyz $O/al12/hypercube_noise/coverage_noise.csv")
    c4=$(sub "${N}_al12_04" "$c3" "$PART" 40 3988 08:00:00 "$E" "$JL $T al12/04_naive_ensemble.jl")
    # Al_16
    d1=$(sub "${N}_al16_01" "$AD" "$PART" 40 3988 06:00:00 "$E,BUILD_THREADS=16" "$JL $T al16/01_pinned_ensembles.jl")
    d2=$(sub "${N}_al16_02" "$d1" "$PART" 40 3988 08:00:00 "$E" "$JL $T al16/02_bands.jl")
    # the four-panel figure: the Al_12 band cache (31 native Hessians, each member at its own geometry), then the plot
    c5=$(sub "${N}_four_panel_cache" "$c4" "$PART" 40 3988 04:00:00 "$E" "$JL $T four_panel/build_bands_cache_Al_12.jl")
    f4=$(sub "${N}_four_panel" "$c5:$d2" "$PART" 1 8000 00:30:00 "$E" "$JL four_panel/plot_four_panel.jl")
    # RLS vs constrained RLS, 200 K NPT of both (same seed: common random numbers)
    local NPT="TEMPS=200,SEED_MODE=same,MD_SEED=1234,MODELDIR=$AM/Al_12_4_6A_2_"
    # 20 threads with two CPUs' memory each: six 5×5×5 undotted Hessians (500 atoms, 1.6 GB per thread buffer)
    r1=$(sub "${N}_rls_01" "$AD" "$PART" 20 7976 12:00:00 "$E" "$JL $T rls_npt/01_constrained_rls.jl")
    n1=$(sub "${N}_npt_rls" "$AD" "$PART" 4 3988 24:00:00 \
         "$E,$NPT,THETA_FILE=$AM/Al_12_4_6A_2_/lin_params.csv,TAG=rls,OUTDIR=$O/rls_npt/npt/rls" "$JL $T rls_npt/02_npt.jl")
    n2=$(sub "${N}_npt_con" "$r1" "$PART" 4 3988 24:00:00 \
         "$E,$NPT,THETA_FILE=$O/rls_npt/constrained/theta_rls_constrained.csv,TAG=constrained,OUTDIR=$O/rls_npt/npt/constrained" \
         "$JL $T rls_npt/02_npt.jl")
    f2=$(sub "${N}_fig_200K" "$n1:$n2" "$PART" 8 3988 02:00:00 "$E" \
         "for m in rls constrained; do $JL rls_npt/03_rdf.jl $O/rls_npt/npt/\$m 200; done && $JL $T rls_npt/04_fig_200K.jl")
    # MACE at a0 = 4.032 Å: the constrained mean model, then constrained vs unconstrained POPS
    local ME="$E,THREADS=16,OMP_NUM_THREADS=16"
    m1=$(sub "${N}_mace_fit" "$MD" "$PART" 16 2000 12:00:00 "$ME" "$PY mace/1_constraints.py && $PY mace/2_design.py && $JL mace/3_qp.jl")
    m2=$(sub "${N}_mace_compare" "$m1" "$PART" 16 2000 06:00:00 "$ME" "$PY mace/4_compare.py && $PY mace/5_elastic_matscipy.py")
    p1=$(sub "${N}_pops_rows" "$MD" "$PART" 16 2000 12:00:00 "$ME" "$PY mace/pops/p1_rows.py")
    p2=$(sub "${N}_pops_cloud" "$p1:$m1" "$PART" 16 2000 06:00:00 "$ME" "$PY mace/pops/p1b_export_csv.py && $JL $T mace/pops/p2_cloud.jl")
    p3=$(sub "${N}_pops_sample" "$p2" "$PART" 4 2000 02:00:00 "$ME" "$JL mace/pops/p3_sample.jl")
    p4=$(sub "${N}_pops_eval" "$p3" "$PART" 16 2000 06:00:00 "$ME" \
         "$PY mace/pops/p4_evaluate.py && $PY mace/pops/p5_figures.py && $PY mace/pops/p6_maxdev_vs_error.py")
    echo "$N: Al_12 $c1→$c2→$c3(+noise $c3n)→$c4→$c5 | Al_16 $d1→$d2 | four-panel $f4 | RLS $r1, NPT $n1 $n2 → fig $f2"
    echo "$N: MACE $m1→$m2 | POPS $p1→$p2→$p3→$p4      logs: $LOG/"
}

fresh_route() {
    if [ -e "$REPO/repro/$1" ] && [ "${FORCE:-0}" != 1 ]; then
        echo "repro/$1 already exists; FORCE=1 to rerun into it"; exit 1
    fi
    mkdir -p "$REPO/repro/$1" "$LOG"
}

saved() {
    fresh_route saved
    route saved "" "" "$REPO/models" "$REPO/mace/models/Al_small_stagetwo.model"
}

refit() {
    fresh_route refit
    local O=$REPO/repro/refit fit train
    fit=$(sub refit_ace_fit "" "$PART" 16 8000 06:00:00 "OUT=$O,ACE_MODELS=$O/models,NPROCS=16" "$JL ace_models/fit_models.jl 12 16")
    train=$(sub refit_mace_train "" "$PART" 16 2000 24:00:00 "OUT=$O,PY=$PY" "bash mace/train.sh")
    echo "refit: ACE models $fit, MACE training $train"
    route refit "$fit" "$train" "$O/models" "$O/mace/models/Al_small_stagetwo.model"
}

case "${1:-help}" in
setup)
    command -v julia >/dev/null || { echo "needs julia 1.11 on PATH"; exit 1; }
    julia --version | grep -q "1\.11\." || echo "warning: the Manifest was resolved with julia 1.11.8; found $(julia --version)"
    python3 -m venv "$REPO/.venv"
    "$PY" -m pip install --upgrade pip==24.2
    "$PY" -m pip install -r requirements.txt
    $JL -e 'using Pkg; Pkg.Registry.add("General"); Pkg.instantiate(); Pkg.precompile(); using ACEWorkflow; println("Julia environment ready")'
    ;;
data)
    SRC=${2:-$REPO/../physics-informed-priors-repro}
    while read -r sum path; do
        [ -f "$path" ] && continue
        mkdir -p "$(dirname "$path")"
        ln "$SRC/$path" "$path" 2>/dev/null || cp "$SRC/$path" "$path"     # hardlink on one filesystem, else copy
    done < data/tierB.sha256
    sha256sum -c data/tierB.sha256
    ;;
replot)
    need_env
    O=$REPO/repro/replot; mkdir -p "$O"
    OUT=$O SRC12=$REPO/published/al12/naive/bands_four_panel_Al_12.jls SRC16=$REPO/published/al16/bands/bands_two_ensembles.jls \
        $JL four_panel/plot_four_panel.jl
    OUT=$O NPT=$REPO/published/rls_npt/npt $JL -t "${THREADS:-8}" rls_npt/04_fig_200K.jl
    OUT=$O "$PY" mace/4_compare.py --replot
    OUT=$O "$PY" mace/pops/p5_figures.py --replot
    OUT=$O "$PY" mace/pops/p6_maxdev_vs_error.py --replot
    echo "figures → $O/"
    ;;
saved)  need_env; need_data; saved ;;
refit)  need_env; need_data; refit ;;
all)    need_env; need_data; saved; refit ;;
verify)
    need_env
    R=${2:?usage: bash run.sh verify saved|refit}
    { $JL verify/verify.jl "$R"; echo; "$PY" verify/verify_mace.py "$R"; } | tee "$REPO/repro/$R/VERIFY.md"
    ;;
*)
    sed -n 2,17p "$0"; exit 1 ;;
esac
