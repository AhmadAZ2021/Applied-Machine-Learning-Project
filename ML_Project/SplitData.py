import os
import shutil
import random

def split_yolo_dataset(image_dir, output_dir, train_ratio=0.8, test_ratio=0.15, val_ratio=0.05):
    assert train_ratio + test_ratio + val_ratio == 1.0, "Ratios must sum to 1"

    
    valid_exts = ('.jpg', '.jpeg', '.png')
    all_files = os.listdir(image_dir)
    image_files = [f for f in all_files if f.lower().endswith(valid_exts)]

 
    labeled_images = []
    unlabeled_images = []

    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        if label_file in all_files:
            labeled_images.append(img_file)
        else:
            unlabeled_images.append(img_file)

    unlabeled_dir = os.path.join(output_dir, 'unlabeled')
    os.makedirs(unlabeled_dir, exist_ok=True)
    for img_file in unlabeled_images:
        shutil.copy2(os.path.join(image_dir, img_file), os.path.join(unlabeled_dir, img_file))
    print(f"Moved {len(unlabeled_images)} unlabeled images to: {unlabeled_dir}")

 
    random.shuffle(labeled_images)
    total = len(labeled_images)
    train_end = int(train_ratio * total)
    test_end = train_end + int(test_ratio * total)

    train_files = labeled_images[:train_end]
    test_files = labeled_images[train_end:test_end]
    val_files = labeled_images[test_end:]

 
    def copy_pair(file_list, subset):
        img_dst = os.path.join(output_dir, subset, 'images')
        lbl_dst = os.path.join(output_dir, subset, 'labels')
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        for img_file in file_list:
            shutil.copy2(os.path.join(image_dir, img_file), os.path.join(img_dst, img_file))
            label_file = os.path.splitext(img_file)[0] + '.txt'
            shutil.copy2(os.path.join(image_dir, label_file), os.path.join(lbl_dst, label_file))


    copy_pair(train_files, 'train')
    copy_pair(test_files, 'test')
    copy_pair(val_files, 'val')

    print("YOLO dataset split completed.")
    print(f"Train: {len(train_files)}, Test: {len(test_files)}, Val: {len(val_files)}")

# Example usage
split_yolo_dataset(r'<Images Folder Path>')
