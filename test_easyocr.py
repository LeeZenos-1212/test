# EasyOCR极速版（单张＜1秒，2张＜2秒）
import easyocr
import time
from pathlib import Path

reader = easyocr.Reader(['ch_sim'], gpu=False, verbose=False)
img_dir = Path(__file__).parent / "images"
img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in ['.jpg','png']]

start = time.time()
for img in img_files:
    res = reader.readtext(str(img), detail=0)
    print(f"结果：{res}，耗时：{round(time.time()-start,2)}秒")