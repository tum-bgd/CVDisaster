from pathlib import Path
import filetype

# RFC image file extensions supported by TensorFlow
img_exts = {"png", "jpg", "gif", "bmp"}
# path = Path("../CVIAN/00_SVI/0_MinorDamage")
# path = Path("../CVIAN/00_SVI/1_ModerateDamage")
# path = Path("../CVIAN/00_SVI/2_SevereDamage")
# path = Path("../CVIAN/01_Satellite/0_MinorDamage")
# path = Path("../CVIAN/01_Satellite/1_ModerateDamage")
path = Path("../CVIAN/01_Satellite/2_SevereDamage")
for file in path.iterdir():
    if file.is_dir():
        continue
    ext = filetype.guess_extension(file)
    if ext is None:
        print(f"'{file}': extension cannot be guessed from content")
    elif ext not in img_exts:
        print(f"'{file}': not a supported image file")

# has thumb.db. Deletion not necessary.
