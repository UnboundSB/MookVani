content = open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'r', encoding='utf-8').read()

# Fix ISIGN sequential method - same key discard problem
old = '''            while cap.isOpened():
                # Responsive key checking
                key = cv2.waitKey(1) & 0xFF

                # Throttle frame advancement
                now = time.time()
                if now < next_frame_time:
                    continue
                next_frame_time += frame_interval

                ret, frame = cap.read()
                if not ret:
                    break

                current_ts = global_timestamp + (word_frame_idx / fps)

                # Draw alerting word (big red flashing)
                flash_on = draw_alert_word(frame, word, flash_on)

                cv2.putText(frame, f"Sentence: {ent['sentence']}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
                cv2.putText(frame, f"Word {entry_idx + 1}/{len(entries)}  Frame {word_frame_idx}/{word_total}  {current_ts:.2f}s", (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "M = mark  SPACE = skip word  N = next word  R = redo  Q = quit", (30, 660),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

                cv2.imshow(window, frame)

                if cooldown > 0:
                    cooldown -= 1
                    continue

                if key in (ord('m'), ord('M')):'''

new = '''            while cap.isOpened():
                # Read key first - non-blocking, NEVER discarded
                key = cv2.waitKey(1) & 0xFF

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                current_ts = global_timestamp + (word_frame_idx / fps)

                # Process key immediately with current frame data (keys are never discarded)
                if cooldown > 0:
                    cooldown -= 1
                else:
                    if key in (ord('m'), ord('M')):
                        self.db.append(
                            sentence_hash=hash_sentence(ent["sentence"]),
                            sentence=ent["sentence"],
                            sentence_id=ent["sentence_id"],
                            sentence_video=word_video_path,
                            word=word,
                            word_video=word_video_path,
                            frame_index=global_frame_idx,
                            timestamp=current_ts,
                            accepted=1,
                        )
                        self.marks_in_current_sentence += 1
                        print(f"[MARK] frame {global_frame_idx} ({current_ts:.2f}s) -> '{word}'")
                        cooldown = 15
                    elif key in (ord(' '), ord('s'), ord('S')):
                        # Skip this word — break out of current word video loop, move to next word
                        print(f"[SKIP] frame {global_frame_idx} ({current_ts:.2f}s) -> '{word}' not marked")
                        skipped = True
                        # If this is the last word, advance to next sentence immediately
                        if entry_idx + 1 >= len(entries):
                            cap.release()
                            for c in caps[entry_idx + 1:]:
                                if c is not None:
                                    c.release()
                            self._advance_next()
                            return
                        break
                    elif key in (ord('n'), ord('N')):
                        pass
                    elif key in (ord('r'), ord('R')):
                        for c in caps:
                            if c is not None:
                                c.release()
                        self.stage = "words"
                        self.db.checkpoint(self.current_pos, ent["sentence_id"], "words")
                        return
                    elif key in (ord('q'), ord('Q'), 27):
                        for c in caps:
                            if c is not None:
                                c.release()
                        cv2.destroyAllWindows()
                        print("[INFO] Quit by user")
                        sys.exit(0)

                # Draw alerting word (big red flashing)
                flash_on = draw_alert_word(frame, word, flash_on)

                cv2.putText(frame, f"Sentence: {ent['sentence']}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
                cv2.putText(frame, f"Word {entry_idx + 1}/{len(entries)}  Frame {word_frame_idx}/{word_total}  {current_ts:.2f}s", (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "M = mark  SPACE = skip word  N = next word  R = redo  Q = quit", (30, 660),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

                cv2.imshow(window, frame)

                # Throttle: sleep if ahead of schedule (never discard keys)
                now = time.time()
                if now < next_frame_time:
                    time.sleep(next_frame_time - now)
                next_frame_time += frame_interval

                word_frame_idx += 1
                global_frame_idx += 1'''

if old in content:
    content = content.replace(old, new, 1)
    open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'w', encoding='utf-8').write(content)
    print("SUCCESS: Fixed ISIGN key handling")
else:
    print("ERROR: ISIGN block not found")
    # Debug
    import re
    m = re.search(r'while cap\.isOpened\(\):', content)
    if m:
        # Find second occurrence
        matches = list(re.finditer(r'while cap\.isOpened\(\):', content))
        print(f"Found {len(matches)} while cap.isOpened() loops")
        for i, mm in enumerate(matches):
            print(f"  Loop {i} at position {mm.start()}")
            print(f"    Context: {repr(content[mm.start():mm.start()+80])}")
