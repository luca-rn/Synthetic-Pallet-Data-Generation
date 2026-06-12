import albumentations as A
import cv2
from pathlib import Path

transform = A.Compose([
    A.GaussNoise(var_limit=(10, 50), p=0.5),
    A.ISONoise(p=0.3),           # camera sensor noise
    A.MultiplicativeNoise(p=0.3), # speckle-style
    A.Blur(blur_limit=3, p=0.2),
])

for path in Path("/tmp/output").glob("rgb_*.png"):
    img = cv2.imread(str(path))
    augmented = transform(image=img)["image"]
    cv2.imwrite(str(path.parent / f"aug_{path.name}"), augmented)