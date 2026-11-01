content = open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'r', encoding='utf-8').read()

# Remove the orphaned leftover mark block at lines 607-620
old_orphan = '''            next_frame_time += frame_interval
                self.db.append(
                    sentence_hash=hash_sentence(ent["sentence"]),
                    sentence=ent["sentence"],
                    sentence_id=ent["sentence_id"],
                    sentence_video=video_path,
                    word=current_word,
                    word_video=current_word_video,
                    frame_index=fidx,
                    timestamp=ts,
                    accepted=1,
                )
                self.marks_in_current_sentence += 1
                print(f"[MARK] frame {fidx} ({ts:.2f}s) -> '{current_word}'")
                cooldown = 15  # ignore keys for ~15 frames to prevent double-mark
            elif key in (ord(' '), ord('s'), ord('S')):'''

new = '''            next_frame_time += frame_interval
            elif key in (ord(' '), ord('s'), ord('S')):'''

if old_orphan in content:
    content = content.replace(old_orphan, new, 1)
    open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'w', encoding='utf-8').write(content)
    print("SUCCESS: Removed orphaned code")
else:
    print("ERROR: orphan block not found")
    # Show what's around line 605-607
    lines = content.split('\n')
    for i, line in enumerate(lines[603:610], start=604):
        print(f"{i}: {repr(line)}")
