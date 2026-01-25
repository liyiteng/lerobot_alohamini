from datasets import load_dataset

# 1. Read parquet directly as a Dataset (train split)
ds = load_dataset(
    "parquet",
    data_files="./episode_000003.parquet",
    split="train"               # specify split to return Dataset (not DatasetDict)
)

# 2. Inspect columns
print("columns:", ds.column_names)

# 3. Print a few rows to confirm
df = ds.to_pandas()
print(df.head())

# 4. Use a loose threshold to filter timestamps near 8.63s
targ = 8.63
tol  = 1                    # 1 ms tolerance
ds_8_63 = ds.filter(lambda ex: abs(ex["timestamp"] - targ) < tol)

print(f"Matched rows: {ds_8_63.num_rows}")
for ex in ds_8_63:
    print(ex)

# # -- OR --
# # If you prefer filtering by frame_index (need fps)
# fps = 30
# target_frame = round(targ * fps)   # e.g. round(8.63*30)=259
# ds_frame = ds.filter(lambda ex: ex["frame_index"] == target_frame)
# print(f"By frame_index ({target_frame}):", ds_frame.num_rows)
