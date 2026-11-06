from synthetic.generator import extract_word_features_from_video, load_word_inventory

inv = load_word_inventory()
feats = extract_word_features_from_video('i', inv)
print(f'Extracted {len(feats)} recordings for "i"')
for i, f in enumerate(feats):
    print(f'  Recording {i+1}: shape={f.shape}')