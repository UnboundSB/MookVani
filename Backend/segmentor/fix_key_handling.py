import re

path = r'D:\MookVani\backend\segmentor\data_builder/opencv_app.py'
content = open(path, 'r', encoding='utf-8').read()

# Fix _play_full_with_marking: move key processing before frame read, use time.sleep instead of continue
old_full = '''        flash_on = False
        cooldown = 0  # frames to ignore key presses after a mark
        next_frame_time = time.time()

        while cap.isOpened():
            # Responsive key checking: poll every 1ms so we never miss a keypress
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
                    break

            # Draw alerting word (big red flashing)
            flash_on = draw_alert_word(frame, current_word, flash_on)

            cv2.putText(frame, f"Sentence: {ent['sentence']}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
            cv2.putText(frame, f"Take {take_idx + 1}/{len(takes)}  Frame {fidx}/{total}  {ts:.2f}s", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "M = mark  SPACE = skip word  T = next take  N = next word  R = redo  Q = quit", (30, 660),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

            cv2.imshow(self.window, frame)

            if cooldown > 0:
                cooldown -= 1
                continue

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
                cooldown = 15  # ignore keys for ~15 frames to prevent double-mark
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
                # Next take for same sentence
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
                sys.exit(0)

        cap.release()'''

new_full = '''        flash_on = False
        cooldown = 0  # frames to ignore key presses after a mark
        next_frame_time = time.time()

        while cap.isOpened():
            # Read key first - non-blocking, NEVER discarded
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
                    sys.exit(0)

            # Draw alerting word (big red flashing)
            flash_on = draw_alert_word(frame, current_word, flash_on)

            cv2.putText(frame, f"Sentence: {ent['sentence']}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
            cv2.putText(frame, f"Take {take_idx + 1}/{len(takes)}  Frame {fidx}/{total}  {ts:.2f}s", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "M = mark  SPACE = skip word  T = next take  N = next word  R = redo  Q = quit", (30, 660),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

            cv2.imshow(self.window, frame)

            # Throttle: sleep if ahead of schedule (never discard keys)
            now = time.time()
            if now < next_frame_time:
                time.sleep(next_frame_time - now)
            next_frame_time += frame_interval

        cap.release()'''

if old_full in content:
    content = content.replace(old_full, new_full)
    open(path, 'w', encoding='utf-8').write(content)
    print("Fixed _play_full_with_marking")
else:
    print("ERROR: old_full not found in _play_full_with_marking")
    # Debug: show what's actually there
    import re
    match = re.search(r'flash_on = False.*?cap\.release\(\n', content, re.DOTALL)
    if match:
        print("Found section (first 500 chars):")
        print(match.group(0)[:500])