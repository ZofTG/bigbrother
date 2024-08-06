"""module used to convert png images to base64 entities"""

import os
import base64 as b64
from os.path import dirname, join


# create the dict
icons_dict = {}
path = dirname(dirname(__file__))
icons_path = join(path, "icons")
for name in os.listdir(icons_path):
    filename = join(icons_path, name)
    label = name.split(".")[0].upper()
    with open(filename, "rb") as buf:
        icons_dict[label] = b64.b64encode(buf.read()).decode("utf-8")

# convert to a readable file
keys = list(icons_dict.keys())
file_list = ['"""assets data module"""']
file_list += ["= ".join([i, '"' + v + '"']) for i, v in icons_dict.items()]
file_list += ["ICONS = {" + ",".join(['"' + i + '":' + i for i in keys]) + ",}"]
all_cmd = "__all__=[" + ", ".join([f"'{i}'" for i in keys] + ["'ICONS'"]) + "]"
file_list += [all_cmd]
file = "\n".join(file_list)

# save
assets_file = join(path, "src", "assets.py")
with open(assets_file, "w") as buf:
    buf.write(file)
