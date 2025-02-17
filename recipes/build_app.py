"""script used to generate the IRCam executable"""

#! IMPORTS


import shutil
import subprocess
from os import remove, getcwd
from os.path import join

from PIL import Image

#! FUNCTION


def main():
    """build the exe"""
    root = getcwd()

    # make the icon
    icon_file = "icon.ico"
    Image.open(join(root, "icons", "main.png")).save(icon_file)

    # run command
    msg = [".venv\\Scripts\\python.exe", "pyinstaller", "--name IRCam", "--clean"]
    msg += ['--add-data "assets;assets"']
    msg += ["--noconsole", f"--icon {icon_file}"]
    msg += ["--onefile run.py"]
    subprocess.run(" ".join(msg))

    # move the exe to the home directory
    shutil.copyfile("dist/IRCam.exe", "IRCam.exe")

    # remove the temporary files
    shutil.rmtree("build")
    shutil.rmtree("dist")
    remove(icon_file)
    remove("IRCam.spec")


#! MAIN


if __name__ == "__main__":
    main()
