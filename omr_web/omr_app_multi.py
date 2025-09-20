"""
Streamlit OMR Batch Grader (multi-sheet capable)

Save as: omr_app_multi.py
Run: streamlit run omr_app_multi.py
"""

import streamlit as st
import cv2
import numpy as np
import imutils
import json
import pandas as pd
import tempfile
import os
import shutil
import zipfile
from io import BytesIO
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor, as_completed
from imutils.perspective import four_point_transform
from tqdm import tqdm

st.set_page_config(page_title="OMR Batch Grader", layout="wide")

# -------------------------------------------------------
# OMR logic (adapted from your functions)
# -------------------------------------------------------
def load_answer_key_from_bytes(b):
    return json.loads(b.decode("utf-8"))

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method in ["right-to-left", "bottom-to-top"]:
        reverse = True
    if method in ["top-to-bottom", "bottom-to-top"]:
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)
    )
    return list(cnts)

def find_document_contour(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None

def extract_bubbles(thresh_warp, question_cnt, min_w=18, min_h=18):
    cnts = cv2.findContours(thresh_warp.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bubb_contours = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h) if h>0 else 0
        if w >= min_w and h >= min_h and 0.7 <= ar <= 1.5:
            bubb_contours.append(c)
    if len(bubb_contours) == 0:
        return []

    bubb_contours = sort_contours(bubb_contours, method="top-to-bottom")
    centers = [(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] // 2) for c in bubb_contours]
    miny, maxy = min(centers), max(centers)
    if question_cnt <= 0:
        return []

    bins = np.linspace(miny - 1, maxy + 1, question_cnt + 1)
    rows = [[] for _ in range(question_cnt)]
    for c, cy in zip(bubb_contours, centers):
        idx = np.searchsorted(bins, cy) - 1
        idx = max(0, min(question_cnt - 1, idx))
        rows[idx].append(c)

    grouped = []
    for row in rows:
        if len(row) == 0:
            grouped.append([])
            continue
        row_sorted = sort_contours(row, method="left-to-right")
        grouped.append(row_sorted)
    return grouped

def grade_sheet_image(image_np, answer_key, debug=False):
    """Grade a single image provided as numpy array (BGR). Returns result dict + annotated image (BGR)."""
    orig = image_np.copy()
    # protect against tiny images
    if image_np.shape[0] < 200 or image_np.shape[1] < 200:
        raise RuntimeError("Image too small or invalid")

    ratio = image_np.shape[0] / 700.0
    image = imutils.resize(image_np, height=700)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    docCnt = find_document_contour(edged)
    if docCnt is None:
        # fallback: try adaptive threshold and morphological close to help
        warped = image  # try using full image
        warped = four_point_transform(orig, np.array([[0,0],[orig.shape[1]-1,0],[orig.shape[1]-1,orig.shape[0]-1],[0,orig.shape[0]-1]], dtype="float32"))
    else:
        warped = four_point_transform(orig, docCnt.reshape(4, 2) * ratio)

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # adaptive threshold if lighting uneven
    try:
        thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 10)
    except:
        thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # morphological open to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    num_q = int(answer_key["num_questions"])
    num_choices = int(answer_key["num_choices"])

    grouped = extract_bubbles(thresh, num_q)
    if len(grouped) == 0:
        raise RuntimeError("No bubble contours found. Check sheet layout and parameters.")

    results = []
    annotated = warped.copy()

    for q_idx, row in enumerate(grouped):
        if len(row) == 0:
            results.append({"question": q_idx + 1, "detected": None, "status": "no_bubbles_found"})
            continue

        # take left-most num_choices if too many
        if len(row) > num_choices:
            row = sort_contours(row, method="left-to-right")[:num_choices]

        counts = []
        bubble_boxes = []
        for c in row:
            (x, y, w, h) = cv2.boundingRect(c)
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            counts.append(int(total))
            bubble_boxes.append((x, y, w, h))

        counts_arr = np.array(counts)
        if counts_arr.size == 0:
            results.append({"question": q_idx + 1, "detected": None, "status": "no_bubbles_found"})
            continue

        sorted_idx = np.argsort(-counts_arr)
        top = counts_arr[sorted_idx[0]]
        second = counts_arr[sorted_idx[1]] if len(counts_arr) > 1 else 0

        absolute_min = max(30, int(np.mean(counts_arr) * 0.2))  # adaptive min
        selection = None
        ambiguous = False
        if top < absolute_min:
            selection = None
        elif second == 0 or top > second * 1.25:
            selection = int(sorted_idx[0])
        else:
            selection = int(sorted_idx[0])
            ambiguous = True

        correct_choice = answer_key["answers"][q_idx]
        if selection is None:
            status = "no_mark"
            correct = False
        else:
            correct = (selection == correct_choice)
            status = "correct" if correct else "incorrect"
            if ambiguous:
                status += "_ambiguous"

        if selection is not None and selection < len(bubble_boxes):
            (x, y, w, h) = bubble_boxes[selection]
            color = (0, 255, 0) if correct else (0, 0, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, str(selection), (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        results.append({
            "question": q_idx + 1,
            "detected": selection,
            "correct_choice": correct_choice,
            "is_correct": correct,
            "status": status,
            "counts": "|".join(map(str, counts))
        })

    correct_count = sum(1 for r in results if r.get("is_correct"))
    score = (correct_count / num_q) * 100.0

    return {"results": results, "score": score, "correct_count": correct_count, "total_questions": num_q}, annotated

# -------------------------------------------------------
# Utility: process single file (image or pdf page)
# -------------------------------------------------------
def process_single_sheet_imagebytes(image_bytes, answer_key, basename, out_dir):
    # image_bytes: raw bytes for an image file (jpg/png)
    # returns paths of saved outputs
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Unable to read image data")

    results, annotated = grade_sheet_image(img, answer_key)
    # prepare filenames
    out_img_path = os.path.join(out_dir, f"{basename}_annotated.png")
    out_csv_path = os.path.join(out_dir, f"{basename}_results.csv")
    # save annotated
    cv2.imwrite(out_img_path, annotated)
    # save per-sheet csv
    df = pd.DataFrame(results["results"])
    # add summary row
    summary = {"question": "Score", "detected": "", "correct_choice": f"{results['correct_count']}/{results['total_questions']}",
               "is_correct": "", "status": f"{results['score']:.2f}%","counts": ""}
    df.loc[len(df)] = summary
    df.to_csv(out_csv_path, index=False)
    return {"base": basename, "score": results["score"], "correct_count": results["correct_count"],
            "total_questions": results["total_questions"], "img": out_img_path, "csv": out_csv_path, "results": results}

# -------------------------------------------------------
# Streamlit UI and batch processing
# -------------------------------------------------------
st.title("📚 OMR Batch Grader — Multi-sheet")

st.write("Upload multiple OMR sheets (images or PDFs). Also upload one answer-key JSON file that applies to all sheets.")

uploaded_files = st.file_uploader("Upload OMR sheets (multiple)", accept_multiple_files=True,
                                  type=['jpg','jpeg','png','tif','tiff','pdf'])
key_file = st.file_uploader("Upload Answer Key (JSON) — single file", accept_multiple_files=False, type=['json'])

# optional Poppler path input (if needed on Windows)
poppler_path = st.text_input("Poppler bin path (leave empty if poppler is on PATH)", value="")

if st.button("Start Batch Grading"):
    if not uploaded_files or not key_file:
        st.error("Please upload at least one sheet and the answer key JSON.")
    else:
        # Load answer key
        try:
            answer_key = load_answer_key_from_bytes(key_file.read())
        except Exception as e:
            st.error(f"Failed to read answer key JSON: {e}")
            st.stop()

        # create output folder
        tmp_root = tempfile.mkdtemp(prefix="omr_batch_")
        out_dir = os.path.join(tmp_root, "outputs")
        os.makedirs(out_dir, exist_ok=True)

        # prepare list of (image_bytes, basename) to process
        tasks = []
        file_count = 0
        for up in uploaded_files:
            fname = up.name
            lower = fname.lower()
            if lower.endswith(".pdf"):
                # convert pdf pages
                try:
                    if poppler_path.strip():
                        pages = convert_from_path(up, dpi=200, poppler_path=poppler_path.strip())
                    else:
                        # convert_from_path expects path; we need to write to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                            tmp_pdf.write(up.read())
                            tmp_pdf.flush()
                            tmp_pdf_path = tmp_pdf.name
                        pages = convert_from_path(tmp_pdf_path, dpi=200, poppler_path=poppler_path.strip() or None)
                        os.unlink(tmp_pdf_path)
                    for i, page in enumerate(pages):
                        file_count += 1
                        basename = f"{os.path.splitext(fname)[0]}_page{i+1}"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpimg:
                            page.save(tmpimg.name, "JPEG")
                            with open(tmpimg.name, "rb") as f:
                                b = f.read()
                            tasks.append((b, basename))
                            os.unlink(tmpimg.name)
                except Exception as e:
                    st.error(f"PDF conversion failed for {fname}: {e}")
            else:
                # image file
                file_count += 1
                b = up.read()
                basename = os.path.splitext(fname)[0]
                tasks.append((b, basename))

        st.info(f"Prepared {len(tasks)} sheets for grading (from {len(uploaded_files)} uploaded files).")

        # Process in parallel with ThreadPoolExecutor
        results_summary = []
        futures = []
        max_workers = min(6, max(1, os.cpu_count() or 1))
        progress_bar = st.progress(0)
        status_text = st.empty()

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for b, basename in tasks:
                futures.append(ex.submit(process_single_sheet_imagebytes, b, answer_key, basename, out_dir))

            completed = 0
            for fut in as_completed(futures):
                try:
                    info = fut.result()
                    results_summary.append(info)
                except Exception as e:
                    # record failure
                    results_summary.append({"base": "error", "score": None, "error": str(e)})
                completed += 1
                progress_bar.progress(int(100 * completed / len(futures)))
                status_text.text(f"Processed {completed}/{len(futures)} sheets")

        # create aggregated CSV
        agg_rows = []
        for r in results_summary:
            if "error" in r:
                agg_rows.append({"sheet": r.get("base","unknown"), "score": None, "correct_count": None, "total_questions": None, "error": r.get("error")})
            else:
                agg_rows.append({"sheet": r["base"], "score": r["score"], "correct_count": r["correct_count"], "total_questions": r["total_questions"]})
        agg_df = pd.DataFrame(agg_rows)
        agg_csv_path = os.path.join(out_dir, "aggregate_results.csv")
        agg_df.to_csv(agg_csv_path, index=False)

        # create a zip of outputs
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # add per-sheet outputs
            for r in results_summary:
                if "img" in r and os.path.exists(r["img"]):
                    zf.write(r["img"], arcname=os.path.basename(r["img"]))
                if "csv" in r and os.path.exists(r["csv"]):
                    zf.write(r["csv"], arcname=os.path.basename(r["csv"]))
            # add aggregated CSV
            zf.write(agg_csv_path, arcname=os.path.basename(agg_csv_path))
        zip_buffer.seek(0)

        st.success("Batch grading completed!")
        st.write("## Summary")
        st.dataframe(agg_df)

        st.download_button("📥 Download All Results (ZIP)", zip_buffer.getvalue(), file_name="omr_results.zip", mime="application/zip")

        # show thumbnails for first few annotated images and allow per-file download
        st.write("## Annotated sheets (preview)")
        for r in results_summary[:10]:
            if "img" in r and os.path.exists(r["img"]):
                st.image(r["img"], caption=f"{r['base']} — score: {r.get('score')}", use_column_width=False)
                with open(r["img"], "rb") as f:
                    btn = st.download_button(f"Download {os.path.basename(r['img'])}", f.read(), file_name=os.path.basename(r['img']))

        # cleanup temp files when desired
        if st.button("Clear temporary outputs"):
            try:
                shutil.rmtree(tmp_root)
                st.success("Temporary outputs removed.")
            except Exception as e:
                st.error(f"Cleanup failed: {e}")
