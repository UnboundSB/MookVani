content = open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'r', encoding='utf-8').read()
lines = content.split('\n')

# Remove the orphaned elif key == ord('t') block at line 607 (0-indexed: 606)
# Find it: after "next_frame_time += frame_interval" there should be "elif key == ord('t')"
start_idx = None
for i, line in enumerate(lines):
    if 'next_frame_time += frame_interval' in line and i > 600:
        # Check next non-empty line
        for j in range(i + 1, min(i + 5, len(lines))):
            if lines[j].strip() and "elif key == ord('t')" in lines[j]:
                start_idx = j
                break
        break

if start_idx is None:
    print("ERROR: Could not find orphaned elif key == ord('t')")
else:
    # Find end: the line with "self._play_full_with_marking(ent, entries, takes, next_idx, is_real=True)"
    # followed by "                return"
    end_idx = None
    for i in range(start_idx, len(lines)):
        if "self._play_full_with_marking(ent, entries, takes, next_idx, is_real=True)" in lines[i]:
            # The return is on the next line
            end_idx = i + 2  # include the return line
            break
    
    if end_idx is None:
        print("ERROR: Could not find end of orphaned block")
    else:
        # Also check if there's a blank line after
        del lines[start_idx:end_idx]
        content = '\n'.join(lines)
        open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'w', encoding='utf-8').write(content)
        print(f"SUCCESS: Removed orphaned elif key == ord('t') block (lines {start_idx+1} to {end_idx})")
