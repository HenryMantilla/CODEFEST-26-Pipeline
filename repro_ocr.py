"""
repro_ocr.py — isolate the ResizeImgError with a full traceback, and test
whether pinning Det.lang_type (which rapidocr_latin_params() in
build_index.py currently never sets) fixes it.

build_index.py's own error handling does `str(exc)[:160]`, and this
exception's str() is empty, so the pipeline log can never show more than
"ResizeImgError:" no matter how the batch run is configured. This script
bypasses all of that and calls RapidOCR directly on one real page, so the
traceback points at the actual line inside rapidocr that raises it.

Usage:
    python repro_ocr.py "CORPUS CODEFEST AD ASTRA 2026/.../CSIS_nasa-act1958.pdf"
    python repro_ocr.py <path.pdf> --page 0 --dpi 300
    python repro_ocr.py <path.pdf> --dpi 200      # test the DPI theory
"""
import argparse
import traceback

import fitz
import numpy as np


def rasterize(path: str, page_no: int, dpi: int):
    doc = fitz.open(path)
    page = doc[page_no]
    pixmap = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n)
    if pixmap.n == 4:
        img = img[:, :, :3]
    img = img[:, :, ::-1].copy()   # RGB -> BGR, same as build_index.py's _image()
    doc.close()
    return img


def try_config(name: str, img, params: dict | None):
    from rapidocr import RapidOCR
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    if params:
        for k, v in params.items():
            print(f"  {k} = {v}")
    try:
        reader = RapidOCR() if params is None else RapidOCR(params=params)
    except Exception:
        print("  INIT FAILED:")
        traceback.print_exc()
        return False
    try:
        result = reader(img)
        n = len(getattr(result, "txts", []) or [])
        print(f"  OK -- {n} text region(s) detected")
        return True
    except Exception:
        print("  CALL FAILED -- full traceback:")
        traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    img = rasterize(args.pdf, args.page, args.dpi)
    print(f"rasterized page {args.page} at {args.dpi} dpi")
    print(f"  shape={img.shape}  dtype={img.dtype}  "
          f"contiguous={img.flags['C_CONTIGUOUS']}  "
          f"min={img.min()}  max={img.max()}")

    from rapidocr import OCRVersion, LangRec, LangDet, ModelType

    results = {}

    # 0. bare defaults -- reproduces the batch run's final fallback exactly
    results["bare defaults"] = try_config("0. bare RapidOCR() -- no params", img, None)

    # 1. current build_index.py pinning: Rec fully pinned, Det.lang_type UNSET
    results["current pinning"] = try_config(
        "1. current build_index.py pinning (Det.lang_type unset)", img, {
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.lang_type": LangRec.LATIN,
            "Rec.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.SERVER,
        })

    # 2. same, but with Det.lang_type explicitly pinned too -- the gap this
    #    script exists to test
    for det_lang_name, det_lang in (("LATIN", getattr(LangDet, "LATIN", None)),
                                    ("EN", getattr(LangDet, "EN", None))):
        if det_lang is None:
            continue
        results[f"Det.lang_type={det_lang_name}"] = try_config(
            f"2. + Det.lang_type={det_lang_name} pinned", img, {
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.lang_type": LangRec.LATIN,
                "Rec.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Det.lang_type": det_lang,
                "Det.model_type": ModelType.SERVER,
            })

    # 3. Det on MOBILE instead of SERVER, in case SERVER is the mismatched
    #    piece rather than the missing lang_type
    results["Det.model_type=MOBILE"] = try_config(
        "3. Det.model_type=MOBILE (drop the SERVER pin)", img, {
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.lang_type": LangRec.LATIN,
            "Rec.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE,
        })

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
    print(
        "\nWhichever config says OK above, its exact params dict is what "
        "should replace rapidocr_latin_params()'s return value in "
        "build_index.py. If EVERY config fails, paste the full traceback "
        "from '1. current build_index.py pinning' -- that is the one your "
        "batch run is actually trying to use, and the raise site inside "
        "rapidocr's own source is what to search "
        "https://github.com/RapidAI/RapidOCR/issues for.")


if __name__ == "__main__":
    main()