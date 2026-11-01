content = open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'r', encoding='utf-8').read()
lines = content.split('\n')

# Remove orphaned elif blocks at lines 607-618 (0-indexed: 606-617)
# These are: elif n, elif r, elif q - all duplicates
start_idx = None
for i, line in enumerate(lines):
    if i >= 606 and "elif key in (ord('n'), ord('N')):" in line:
        start_idx = i
        break

if start_idx is None:
    print("ERROR: Could not find orphaned elif n block")
else:
    # Find end: the line with sys.exit(0) followed by blank line and cap.release()
    end_idx = None
    for i in range(start_idx, len(lines)):
        if "sys.exit(0)" in lines[i]:
            end_idx = i + 1  # include the sys.exit(0) line
            break
    
    if end_idx is None:
        print("ERROR: Could not find end of orphaned block")
    else:
        del lines[start_idx:end_idx]
        content = '\n'.join(lines)
        open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'w', encoding='utf-8').write(content)
        print(f"SUCCESS: Removed orphaned elifs n/r/q (lines {start_idx+1} to {end_idx})")
