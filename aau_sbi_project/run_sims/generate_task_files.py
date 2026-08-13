# make_taskfile_82.py
import os
import pickle

perturber_dir = "/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/perturbers/pert_813_mocks"
sims_dir = "/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/sims/sims_pert_813_mocks"
chunk_size = 100
out_taskfile = "taskfile_813.txt"

pert_files = sorted(f for f in os.listdir(perturber_dir) if f.endswith(".pkl"))

# Existing output files, so we can figure out which (pert_file, start, end) chunks
# have already been run.
existing_outputs = set(os.listdir(sims_dir)) if os.path.isdir(sims_dir) else set()


def output_exists(pert_file, start, end):
    """
    Return True if the output for this chunk already exists in sims_dir.

    Matches the pattern:
        perturbers_batch_0000_streams_00000_00100.h5
    i.e. <pert_file_stem>_streams_<start:05d>_<end:05d>.h5
    """
    pert_stem = os.path.splitext(pert_file)[0]
    expected_name = f"{pert_stem}_streams_{start:05d}_{end:05d}.h5"
    return expected_name in existing_outputs


total_chunks = 0
skipped_chunks = 0
written_chunks = 0

with open(out_taskfile, "w") as f:
    for pert_file in pert_files:
        with open(os.path.join(perturber_dir, pert_file), "rb") as pf:
            pert_batch = pickle.load(pf)
        n_perturbers = len(pert_batch)

        for start in range(0, n_perturbers, chunk_size):
            end = min(start + chunk_size, n_perturbers)
            total_chunks += 1

            if output_exists(pert_file, start, end):
                skipped_chunks += 1
                continue

            f.write(
                f"cd /expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims && "
                f"python -u simulate_worker_combined.py {pert_file} {start} {end} "
                f"&> logs/task_${{DISBATCH_TASKID_ZP}}.log\n"
            )
            written_chunks += 1

print(
    f"Wrote {written_chunks} remaining tasks (skipped {skipped_chunks} already-completed "
    f"out of {total_chunks} total chunks) across {len(pert_files)} perturber files -> {out_taskfile}"
)