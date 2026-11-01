content = open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'r', encoding='utf-8').read()

# Replace just the problematic block: key read + discard on throttle
old = '''            # Responsive key checking: poll every 1ms so we never miss a keypress
            key = cv2.waitKey(1) & 0xFF

            # Throttle frame advancement to maintain slowdown speed
            now = time.time()
            if now < next_frame_time:
                continue
            next_frame_time += frame_interval

            ret, frame = cap.read()
            if not ret:
                break
            fidx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            ts = fidx / fps

            current_word = ""
            current_word_video = ""
            for w, start_f, end_f, vpath in word_frame_map:
                if start_f <= fidx < end_f:
                    current_word = w
                    current_word_video = vpath if vpath else ""
                    break'''

new = '''            # Read key first - non-blocking, NEVER discarded
            key = cv2.waitKey(1) & 0xFF

            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            fidx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            ts = fidx / fps

            current_word = ""
            current_word_video = ""
            for w, start_f, end_f, vpath in word_frame_map:
                if start_f <= fidx < end_f:
                    current_word = w
                    current_word_video = vpath if vpath else ""
                    break

            # Process key immediately with current frame data (keys are never discarded)
            if cooldown > 0:
                cooldown -= 1
            else:
                if key in (ord('m'), ord('M')):
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
                    cooldown = 15
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
                elif key == ord('t') or key == ord('T'):
                    cap.release()
                    next_idx = (take_idx + 1) % len(takes)
                    print(f"[NEXT TAKE] Switching to take {next_idx + 1}/{len(takes)}")
                    self._play_full_with_marking(ent, entries, takes, next_idx, is_real=True)
                    return
                elif key in (ord('n'), ord('N')):
                    pass
                elif key in (ord('r'), ord('R')):
                    cap.release()
                    self.stage = "words"
                    self.db.checkpoint(self.current_pos, ent["sentence_id"], "words")
                    return
                elif key in (ord('q'), ord('Q'), 27):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("[INFO] Quit by user")
                    sys.exit(0)'''

if old in content:
    content = content.replace(old, new, 1)
    # Now fix the end of the loop: replace "if cooldown > 0: cooldown -= 1; continue" with nothing
    # and replace the end throttle with time.sleep
    old2 = '''            cv2.imshow(self.window, frame)

            if cooldown > 0:
                cooldown -= 1
                continue

            if key in (ord('m'), ord('M')):'''
    new2 = '''            cv2.imshow(self.window, frame)

            # Throttle: sleep if ahead of schedule (never discard keys)
            now = time.time()
            if now < next_frame_time:
                time.sleep(next_frame_time - now)
            next_frame_time += frame_interval'''
    
    if old2 in content:
        content = content.replace(old2, new2, 1)
        open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'w', encoding='utf-8').write(content)
        print("SUCCESS: Fixed key handling")
    else:
        print("ERROR: old2 not found")
        print("Looking for the end section...")
        import re
        # Find what's actually there
        m = re.search(r'cv2\.imshow\(self\.window, frame\)\n\n            if cooldown > 0:', content)
        if m:
            print("Found:", repr(m.group(0)[:200]))
else:
    print("ERROR: old block not found")
    # Show what we're looking for vs what's there
    import re
    # Find the while loop section
    m = re.search(r'while cap\.isOpened\(\):', content)
    if m:
        print("Found while loop at position", m.start())
        print("Next 100 chars:", repr(content[m.start():m.start()+100]))
