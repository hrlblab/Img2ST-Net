def get_train_val_paths(root_path, test_slide_name):
    from pathlib import Path

    root_path = Path(root_path)
    all_npy_files = list(root_path.glob("*.npy"))

    # ✅ Automatically remove the extension from the test slide name
    test_slide_name = Path(test_slide_name).stem
    train_paths = []
    val_paths = []

    for npy_path in all_npy_files:
        slide_name = npy_path.stem
        if slide_name == test_slide_name:
            val_paths.append(str(npy_path))
        else:
            train_paths.append(str(npy_path))

    # Raise an error if the specified test slide is not found
    if not val_paths:
        raise ValueError(f"Test slide '{test_slide_name}' was not found in path '{root_path}'.")

    print(f"[INFO] Found {len(train_paths)} training .npy files")
    print(f"[INFO] Validation set: {val_paths[0]}")

    return train_paths, val_paths
