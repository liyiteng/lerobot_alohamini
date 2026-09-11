# Dataset integrity and interrupted saves

Check a local dataset, including full video decoding and safety sidecars:

```bash
python scripts/check_lerobot_dataset_integrity.py \\
  --dataset.root /path/to/dataset --decode-videos --output-json /path/to/report.json
```

Add `--fail-on-warnings` when safety-log gaps must produce a nonzero exit status.

Repair unambiguous metadata/index and video gaps into a new directory:

```bash
python scripts/check_lerobot_dataset_integrity.py \\
  --dataset.root /path/to/dataset --repair-output /path/to/repaired_dataset
```

The source is unchanged. Repairs retain safety events, timestamps and local frame
indices, and remap episode identifiers in both filenames and records. Missing
frames are not synthesized. Ambiguous damage is rejected.

`valid` reports structural integrity. `training_review` is `required` when errors
or safety warnings are present; otherwise it remains `not_assessed`. Neither
value certifies training suitability. Inspect protection events and split capture
discontinuities before training policies that use contiguous action chunks.

Before saving an episode, the writer stores its numeric data and image paths in
`meta/recovery/episode_XXXXXX.json`. Successful saves remove that snapshot.
Failures preserve the in-memory episode and retain source images until commit.
After a partial commit, further appends and repeated saves are refused to prevent
duplicate rows. Preserve the dataset and recovery files for offline recovery;
the integrity repair command does not automatically replay a partial commit.

Insufficient disk space can also prevent the recovery snapshot from being written.
Streaming encoder buffers, power loss and forced process termination are not
covered by this recovery snapshot. Never delete recovery files merely to bypass
an integrity error.
