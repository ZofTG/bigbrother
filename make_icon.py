"""script used to generate the app icon"""

from PIL import Image
from os.path import join

if __name__ == "__main__":
    Image.open(join("icons", "main.png")).save("BigBrother.ico")
