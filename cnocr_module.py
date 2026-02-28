import cnocr
import cv2
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

# ===================== LOG SETUP =====================
def setup_logger() -> logging.Logger:
    log_dir = Path(__file__).parent / "ocr_logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"book_ocr_{time.strftime('%Y%m%d')}.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger("BookOCR")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()

# ===================== CONFIG =====================
CPU_THREADS = 4
OUTPUT_FORMAT = "json"
DEFAULT_IMG_DIR = Path(__file__).parent / "book_images"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "ocr_results"

# ===================== MODEL INIT =====================
# 替换cnocr_ocr_module.py中的init_ocr()函数
def init_ocr() -> cnocr.CnOcr:
    """初始化CNOCR模型（图书打印体专属优化）"""
    try:
        os.environ["CNOCR_VERBOSE"] = "0"
        os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
        os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)

        ocr = cnocr.CnOcr(
            model_name="densenet_lite_136-gru",
            cpu_threads=CPU_THREADS,
        )
        logger.info("CNOCR model (book print font) initialized successfully")
        return ocr
    except Exception as e:
        logger.error("Failed to initialize CNOCR model", exc_info=True)
        raise

OCR_INSTANCE = init_ocr()

# ===================== BATCH RECOGNITION =====================
def recognize_book_images(
    img_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, List[str]]:

    img_dir = img_dir or DEFAULT_IMG_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    img_files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in exts)

    if not img_files:
        logger.warning(f"No images found in directory: {img_dir}")
        return {}

    logger.info(f"Start batch recognition. Total images: {len(img_files)}")
    start_time = time.time()
    result = {}

    for idx, img_path in enumerate(img_files, 1):
        name = img_path.name
        logger.info(f"[{idx}/{len(img_files)}] Processing image: {name}")

        try:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            raw = OCR_INSTANCE.ocr(img)
            texts = [item["text"].strip() for item in raw] if raw else ["No text detected"]
            result[name] = texts
            logger.info(f"[{idx}/{len(img_files)}] Success: {name}")
        except Exception as e:
            result[name] = [f"Recognition failed: {str(e)}"]
            logger.error(f"[{idx}/{len(img_files)}] Failed: {name}", exc_info=True)

    save_recognize_result(result, output_dir)
    total = round(time.time() - start_time, 2)
    logger.info(f"Batch recognition finished. Total time: {total}s")
    #print(result)
    return result

# ===================== SAVE RESULT =====================
def save_recognize_result(result_dict: Dict[str, List[str]], output_dir: Path):
    ts = time.strftime("%Y%m%d_%H%M%S")
    try:
        if OUTPUT_FORMAT == "json":
            out_path = output_dir / f"book_ocr_result_{ts}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4)
        else:
            out_path = output_dir / f"book_ocr_result_{ts}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("===== BOOK OCR RESULTS =====\n")
                for img, lines in result_dict.items():
                    f.write(f"\n[{img}]\n")
                    f.write("\n".join(lines))
                    f.write("\n---\n")
        logger.info(f"Result saved to: {out_path}")
    except Exception as e:
        logger.error("Failed to save recognition result", exc_info=True)

# ===================== SINGLE IMAGE =====================
def recognize_single_book_image(img_path: str) -> List[str]:
    path = Path(img_path)
    logger.info(f"Start single image recognition: {img_path}")

    if not path.exists():
        logger.warning(f"Image not found: {img_path}")
        return ["Image not found"]

    try:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw = OCR_INSTANCE.ocr(img)
        texts = [item["text"].strip() for item in raw] if raw else ["No text detected"]
        logger.info(f"Single image recognition success: {img_path}")
        return texts
    except Exception as e:
        logger.error(f"Single image recognition failed: {img_path}", exc_info=True)
        return [f"Recognition failed: {str(e)}"]

# ===================== TEST =====================
if __name__ == "__main__":
    recognize_book_images()