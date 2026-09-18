import os
import cv2
import glob
import random
import shutil
from collections import defaultdict

def geometric_augment(img, max_shift_frac=0.06, max_rotation_deg=8, allow_flip=True):
    """Pixel-level augmentation: flip / shift / small tilt ONLY. No contrast/brightness/blur —
    those change image quality without adding pose diversity and can hurt landmark detection."""
    h, w = img.shape[:2]
    out = img.copy()

    if allow_flip and random.random() < 0.5:
        out = cv2.flip(out, 1)  # horizontal flip — valid for most 2-hand-symmetric-ish signs;
                                 # NOTE: for signs where left/right hand identity matters semantically,
                                 # flipping could invert meaning. If you notice degraded quality on
                                 # asymmetric signs, set allow_flip=False for this call.

    angle = random.uniform(-max_rotation_deg, max_rotation_deg)
    shift_x = random.uniform(-max_shift_frac, max_shift_frac) * w
    shift_y = random.uniform(-max_shift_frac, max_shift_frac) * h

    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_mat[0, 2] += shift_x
    rot_mat[1, 2] += shift_y

    out = cv2.warpAffine(out, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out

def prepare_augmented_dataset(raw_dir, augmented_dir, holdout_dir, target_count=20, val_split_ratio=0.2):
    """Splits dataset into train/val and augments the train split geometrically."""
    os.makedirs(augmented_dir, exist_ok=True)
    os.makedirs(holdout_dir, exist_ok=True)

    # --- 1. Group real images by class ---
    all_images = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
        all_images.extend(glob.glob(f"{raw_dir}/**/{ext}", recursive=True))

    class_to_images = defaultdict(list)
    for p in all_images:
        class_to_images[os.path.basename(os.path.dirname(p))].append(p)

    print(f"🔍 {len(all_images)} real images across {len(class_to_images)} classes.")

    # --- 2. Split REAL images into train/val FIRST (per class, grouped by source) ---
    train_real, val_real = {}, {}
    for cls, paths in class_to_images.items():
        paths = sorted(paths)
        random.shuffle(paths)
        split_point = max(1, int((1 - val_split_ratio) * len(paths))) if len(paths) > 1 else 1
        train_real[cls] = paths[:split_point]
        val_real[cls] = paths[split_point:] if len(paths) > 1 else []  # single-image classes have NO val

    # --- 3. Copy val images untouched to the holdout dir ---
    for cls, paths in val_real.items():
        save_dir = os.path.join(holdout_dir, cls)
        os.makedirs(save_dir, exist_ok=True)
        for p in paths:
            shutil.copy(p, os.path.join(save_dir, os.path.basename(p)))

    # --- 4. Copy real train images + generate synthetic augmented copies up to target_count ---
    hopeless_classes = []   # classes where augmentation fundamentally can't help (<=3 real images)
    augmented_counts = {}

    for cls, real_paths in train_real.items():
        save_dir = os.path.join(augmented_dir, cls)
        os.makedirs(save_dir, exist_ok=True)

        # copy real train images as-is
        for p in real_paths:
            shutil.copy(p, os.path.join(save_dir, os.path.basename(p)))

        n_real = len(real_paths)
        if n_real <= 3:
            hopeless_classes.append((cls, n_real))

        n_needed = max(0, target_count - n_real)
        for i in range(n_needed):
            src_path = random.choice(real_paths)
            img = cv2.imread(src_path)
            if img is None:
                continue
            aug_img = geometric_augment(img)
            aug_name = f"{os.path.splitext(os.path.basename(src_path))[0]}__aug{i}.jpg"
            cv2.imwrite(os.path.join(save_dir, aug_name), aug_img)

        augmented_counts[cls] = n_real + n_needed

    print(f"\n✅ Augmented training pool ready at {augmented_dir}")
    print(f"   Classes pulled up to target ({target_count}): "
          f"{sum(1 for c in augmented_counts if class_to_images[c] and len(train_real[c]) < target_count)}")
    print(f"   Classes already above target (untouched): "
          f"{sum(1 for c in augmented_counts if len(train_real[c]) >= target_count)}")
    print(f"\n✅ Val holdout ready at {holdout_dir} (100% real images, never augmented)")

    n_no_val = sum(1 for v in val_real.values() if len(v) == 0)
    print(f"\n⚠️ {n_no_val} classes have ZERO val images (only had 1 real image total to begin with) "
          f"— these classes cannot be validated at all, regardless of augmentation.")

    if hopeless_classes:
        print(f"\n⚠️ {len(hopeless_classes)} classes have ≤3 REAL images — augmentation pads the count "
              f"but all synthetic copies still derive from the same 2-3 source photos of one signer's "
              f"one performance. These will likely still overfit/generalize poorly regardless:")
        for cls, n in sorted(hopeless_classes, key=lambda x: x[1]):
            print(f"   {cls}: {n} real image(s)")

if __name__ == "__main__":
    RAW_WORD_LEVEL_DIR = "../data/ISL_CSLRT_Corpus/Frames_Word_Level"
    AUGMENTED_WORD_LEVEL_DIR = "../data/word_level_images_augmented"
    VAL_HOLDOUT_DIR = "../data/word_level_images_val_holdout"
    prepare_augmented_dataset(RAW_WORD_LEVEL_DIR, AUGMENTED_WORD_LEVEL_DIR, VAL_HOLDOUT_DIR)
