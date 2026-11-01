content = open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'r', encoding='utf-8').read()
lines = content.split('\n')

# Find the orphaned block: starts at line 752 (self.db.append() with extra indent)
# and ends at line 794 (sys.exit(0))
# Keep lines 796-797 (word_frame_idx += 1, global_frame_idx += 1)

start_idx = None
for i, line in enumerate(lines):
    # Look for the orphaned self.db.append( at the wrong indentation
    # It should be at line 752 (0-indexed: 751)
    if i >= 750 and 'self.db.append(' in line and 'sentence_hash=hash_sentence' in lines[i] if i + 1 < len(lines) else False:
        # Verify it's the orphaned one (deeply indented)
        if line.startswith('                    self.db.append('):
            start_idx = i
            break

if start_idx is None:
    # Try alternative: find the orphaned block by looking for the pattern
    for i, line in enumerate(lines):
        if '                    self.db.append(' in line and i > 740:
            start_idx = i
            break

if start_idx is None:
    print("ERROR: Could not find start of orphaned ISIGN block")
else:
    # Find end: the line with sys.exit(0) in this orphaned block
    end_idx = None
    for i in range(start_idx, len(lines)):
        if 'sys.exit(0)' in lines[i]:
            end_idx = i + 1  # include this line
            break
    
    if end_idx is None:
        print("ERROR: Could not find end of orphaned ISIGN block")
    else:
        # Verify lines 796-797 (word_frame_idx += 1, global_frame_idx += 1) come after
        print(f"Removing lines {start_idx+1} to {end_idx}")
        del lines[start_idx:end_idx]
        content = '\n'.join(lines)
        open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'w', encoding='utf-8').write(content)
        print("SUCCESS: Removed orphaned ISIGN key block")
