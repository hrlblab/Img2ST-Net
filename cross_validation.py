import numpy as np
from pathlib import Path


def get_train_val_paths(root_path, test_slide_name):
    root_path = Path(root_path)
    all_npy_files = list(root_path.glob("*.npy"))

    test_slide_name = Path(test_slide_name).stem
    train_paths = []
    val_paths = []

    for npy_path in all_npy_files:
        slide_name = npy_path.stem
        if slide_name == test_slide_name:
            val_paths.append(str(npy_path))
        else:
            train_paths.append(str(npy_path))

    if not val_paths:
        raise ValueError(f"Test slide '{test_slide_name}' was not found in path '{root_path}'.")

    return train_paths, val_paths


# def get_train_val_paths(root_path, test_slide_name):
#     from pathlib import Path

#     root_path = Path(root_path)
#     all_npy_files = list(root_path.glob("*.npy"))

#     # ✅ Automatically remove the extension from the test slide name
#     test_slide_name = Path(test_slide_name).stem
#     train_paths = []
#     val_paths = []

#     for npy_path in all_npy_files:
#         slide_name = npy_path.stem
#         if slide_name == test_slide_name:
#             val_paths.append(str(npy_path))
#         else:
#             train_paths.append(str(npy_path))

#     # Raise an error if the specified test slide is not found
#     if not val_paths:
#         raise ValueError(f"Test slide '{test_slide_name}' was not found in path '{root_path}'.")

#     print(f"[INFO] Found {len(train_paths)} training .npy files")
#     print(f"[INFO] Validation set: {val_paths[0]}")

#     return train_paths, val_paths


def get_train_val_data(root_path, test_slide_name, augment_ratio=0.1, seed=42):
    root_path = Path(root_path)
    all_npy_files = list(root_path.glob("*.npy"))

    test_slide_name = Path(test_slide_name).stem
    train_data = []
    val_data = None

    for npy_path in all_npy_files:
        slide_name = npy_path.stem
        data = np.load(str(npy_path), allow_pickle=True)
        if slide_name == test_slide_name:
            val_data = data
        else:
            train_data.append(data)

    if val_data is None:
        raise ValueError(f"Test slide '{test_slide_name}' was not found.")

    train_data = np.concatenate(train_data, axis=0)

    np.random.seed(seed)
    n_aug = int(len(val_data) * augment_ratio)
    aug_idx = np.random.choice(len(val_data), n_aug, replace=False)
    train_data = np.concatenate([train_data, val_data[aug_idx]], axis=0)

    return train_data, val_data
