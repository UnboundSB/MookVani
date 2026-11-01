content = open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'r', encoding='utf-8').read()
lines = content.split('\n')

# Find the duplicate elif block: starts at line 607 (index 606) with "elif key in (ord(' '), ord('s'), ord('S')):"
# and ends before line 625 with "elif key == ord('t')"
# We need to remove lines 607-624 (1-indexed), which is indices 606-623 (0-indexed)
# But keep line 606 ("next_frame_time += frame_interval")

# Find the start: the duplicate elif after next_frame_time += frame_interval
start_idx = None
for i, line in enumerate(lines):
    if 'next_frame_time += frame_interval' in line and i > 600:
        # Check if next line is the duplicate elif
        if i + 1 < len(lines) and "elif key in (ord(' '), ord('s'), ord('S')):" in lines[i + 1]:
            start_idx = i + 1  # start of duplicate block
            break

if start_idx is None:
    print("ERROR: Could not find start of duplicate block")
else:
    # Find the end: the line with "elif key == ord('t')"
    end_idx = None
    for i in range(start_idx, len(lines)):
        if "elif key == ord('t') or key == ord('T" in lines[i]:
            end_idx = i
            break
    
    if end_idx is None:
        print("ERROR: Could not find end of duplicate block")
    else:
        # Remove lines start_idx to end_idx-1 (0-indexed)
        del lines[start_idx:end_idx]
        content = '\n'.join(lines)
        open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'w', encoding='utf-8').write(content)
        print(f"SUCCESS: Removed lines {start_idx+1} to {end_idx} (duplicate elif block)")
