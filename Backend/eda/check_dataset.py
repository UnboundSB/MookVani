import os

dataset_path = r'F:\dataset'
if os.path.exists(dataset_path):
    print(f"Contents of {dataset_path}:")
    for filename in os.listdir(dataset_path):
        filepath = os.path.join(dataset_path, filename)
        size = os.path.getsize(filepath)
        print(f"{filename} - {size} bytes")
else:
    print(f"Path does not exist: {dataset_path}")
