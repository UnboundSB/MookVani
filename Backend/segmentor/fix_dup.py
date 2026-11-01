content = open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'r', encoding='utf-8').read()

# Remove duplicate elif blocks at lines 607-624 (leftover from old code)
old_dup = '''            next_frame_time += frame_interval
            elif key in (ord(' '), ord('s'), ord('S')):
                # Skip this word boundary — jump to next word's start frame
                print(f"[SKIP] frame {fidx} ({ts:.2f}s) -> '{current_word}' not marked")
                current_word_idx = None
                for i, (w, sf, ef, vp) in enumerate(word_frame_map):
                    if w == current_word:
                        current_word_idx = i
                        break
                if current_word_idx is not None and current_word_idx + 1 < len(word_frame_map):
                    next_start_frame = word_frame_map[current_word_idx + 1][1]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_start_frame)
                    print(f"[SKIP] Jumped to next word: {word_frame_map[current_word_idx + 1][0]} at frame {next_start_frame}")
                else:
                    # On last word, skip and advance to next sentence
                    print(f"[SKIP] Last word skipped — advancing to next sentence.")
                    cap.release()
                    self._advance_next()
                    return
            elif key == ord('t') or key == ord('T):'''

new = '''            next_frame_time += frame_interval'''

if old_dup in content:
    content = content.replace(old_dup, new, 1)
    open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'w', encoding='utf-8').write(content)
    print("SUCCESS: Removed duplicate elif block")
else:
    print("ERROR: duplicate block not found")
    # Show what's around line 605-610
    lines = content.split('\n')
    for i, line in enumerate(lines[604:612], start=605):
        print(f"{i}: {repr(line)}")
