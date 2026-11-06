import re

with open(r'D:\MookVani\backend\segmentor\yap_v5\training.py', 'r') as f:
    content = f.read()

content = content.replace('CHECKPOINT_DIR / "best_model.pth"', 'cfg.CHECKPOINT_DIR / "best_model.pth"')

with open(r'D:\MookVani\backend\segmentor\yap_v5\training.py', 'w') as f:
    f.write(content)

print('Done')