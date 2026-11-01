# Debug: show exact content around the section to replace
content = open(r'D:\MookVani\backend\segmentor\data_builder\opencv_app.py', 'r', encoding='utf-8').read()

# Find the while loop in _play_full_with_marking
start = content.find('flash_on = False\n        cooldown = 0')
if start == -1:
    print("Could not find 'flash_on = False'")
else:
    print(f"Found at position {start}")
    # Show the section with repr to see exact whitespace
    section = content[start:start+2000]
    print(repr(section[:500]))
    print("...")
    print(repr(section[-500:]))
