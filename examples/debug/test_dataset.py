from datasets import load_dataset

# 1. 直接把 parquet 读成一个 Dataset（train split）
ds = load_dataset(
    "parquet",
    data_files="./episode_000003.parquet",
    split="train"               # 直接指定 split，返回 Dataset 而非 DatasetDict
)

# 2. 看一下有哪些列
print("columns:", ds.column_names)

# 3. 把前几行打印出来确认一下
df = ds.to_pandas()
print(df.head())

# 4. 用宽松的阈值筛选 timestamp 接近 8.63 秒的行
targ = 8.63
tol  = 1                    # 1 毫秒容差
ds_8_63 = ds.filter(lambda ex: abs(ex["timestamp"] - targ) < tol)

print(f"Matched rows: {ds_8_63.num_rows}")
for ex in ds_8_63:
    print(ex)

# # —— 或者 ——  
# # 如果你更习惯按 frame_index 来选（需要知道 fps）
# fps = 30
# target_frame = round(targ * fps)   # e.g. round(8.63*30)=259
# ds_frame = ds.filter(lambda ex: ex["frame_index"] == target_frame)
# print(f"By frame_index ({target_frame}):", ds_frame.num_rows)
