import os
import pickle

merged_features_train = {}
features_dir = "./incep/train_features2"
part_files_train = sorted([f for f in os.listdir(features_dir) if f.startswith("train_features_part_")])
for part_file in part_files_train:
    try:
        with open(os.path.join(features_dir, part_file), 'rb') as f:
            part_data = pickle.load(f)
            merged_features_train.update(part_data)
    except Exception as e:
        print(f"Error loading {part_file}: {e}")

# Save the merged result
with open("incep/train2014_features2.pkl", "wb") as f:
    pickle.dump(merged_features_train, f)

print(f"Merged {len(part_files_train)} files into 'train2014_features.pkl' with {len(merged_features_train)} features.")
