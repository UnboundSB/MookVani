import os

def print_magic_bytes(filepath):
    with open(filepath, 'rb') as f:
        magic = f.read(8)
        print(f"{filepath}: {magic}")

print_magic_bytes(r'F:\dataset\archive.zip')
print_magic_bytes(r'F:\dataset\isign\iSign-videos_v1.1_part_aa')
