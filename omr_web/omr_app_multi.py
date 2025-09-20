# omr_app_multi.py
"""
Streamlit OMR Grader — adaptive contour mode + ROI/grid fallback
Features:
- Load keys from Excel (.xlsx/.xls) or JSON (flat or workbook with multiple sheets)
- Set selection for multi-sheet Excel (Set A / Set B / subjects)
- Adaptive contour-based bubble detection (PyImageSearch style)
- Detects multiple marks / no marks and returns confidence
- Falls back to ROI/grid (if provided in key) for template-based scanning
- Option to show debug images (intermediate steps) or only final annotated sheet
- Batch processing (multiple images and PDFs), CSV output + annotated images + ZIP download
"""

import streamlit as st
import cv2
import numpy as np
import imutils
import json
import pandas as pd
import tempfile, os, shutil, zipfile
from io import BytesIO
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor, as_completed
from imutils.perspective import four_point_transform
from imutils import contours as imcontours
from typing import Optional, Tuple, Dict, Any, List

st.set_page_config(page_title="OMR — Adaptive + ROI fallback", layout="wide")

# ---------------------------
# Answer key loader (Excel & JSON)
# ---------------------------
def normalize_answer_token(token: Any) -> Optional[int]:
    if token is None:
        return None
    s = str(token).strip()
    if s == "":
        return None
    if s.isdigit():
        return int(s)
    low = s.lower()
    if low in ["a","b","c","d","e","f","g"]:
        return ord(low) - ord("a")
    # try parse last token or letter
    for t in reversed(s.replace(":", " ").replace("-", " ").replace(".", " ").split()):
        if t.isdigit():
            return int(t)
        if t.lower() in ["a","b","c","d","e","f","g"]:
            return ord(t.lower()) - ord("a")
    return None

def load_key_from_excel_bytes(b: bytes) -> Dict[str, Any]:
    """
    Read workbook bytes and return either flat key or dict of sheets.
    Each sheet must contain a column 'Answer' or second column as answers.
    Optionally a sheet may include a top-row ROI specification (roi_y1, roi_y2, roi_x1, roi_x2).
    Returns either:
      - {"num_questions": N, "num_choices": 4, "answers": [...], "roi": [y1,y2,x1,x2] (optional)}
      OR a dict of such entries keyed by sheet name.
    """
    xls = pd.ExcelFile(BytesIO(b))
    all_keys = {}
    for sheet in xls.sheet_names:
        df_raw = pd.read_excel(xls, sheet_name=sheet, header=None, engine="openpyxl")
        # try detect ROI in first few rows
        roi = None
        for r in range(min(5, len(df_raw))):
            row = df_raw.iloc[r].astype(str).str.lower().tolist()
            joined = " ".join(row)
            if "roi" in joined or "roi_y" in joined or "roi_x" in joined:
                if r + 1 < len(df_raw):
                    nxt = df_raw.iloc[r+1].tolist()
                    nums = []
                    for v in nxt:
                        try:
                            nums.append(int(float(v)))
                        except Exception:
                            pass
                    if len(nums) >= 4:
                        roi = nums[:4]
                break
        # header detection
        header_row = None
        for r in range(min(5, len(df_raw))):
            row_low = df_raw.iloc[r].astype(str).str.lower().tolist()
            if any("answer" in str(x) or "ans" in str(x) for x in row_low):
                header_row = r
                break
        if header_row is not None:
            df = pd.read_excel(xls, sheet_name=sheet, header=header_row, engine="openpyxl")
            ans_col = None
            for col in df.columns:
                if "answer" in str(col).lower() or "ans" in str(col).lower():
                    ans_col = col
                    break
            if ans_col is None:
                ans_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            answers_raw = df[ans_col].tolist()
            answers = [normalize_answer_token(a) for a in answers_raw]
        else:
            # fallback -> second column
            answers = []
            for i in range(len(df_raw)):
                if df_raw.shape[1] >= 2:
                    token = df_raw.iloc[i,1]
                    n = normalize_answer_token(token)
                    if n is not None:
                        answers.append(n)
            if len(answers) == 0:
                # try all cells
                for cell in df_raw.values.flatten().tolist():
                    n = normalize_answer_token(cell)
                    if n is not None:
                        answers.append(n)
        key = {"num_questions": len(answers), "num_choices": 4, "answers": answers}
        if roi:
            key["roi"] = roi
        all_keys[sheet] = key
    if len(all_keys) == 1:
        return list(all_keys.values())[0]
    return all_keys

def load_answer_key_file(uploaded_file) -> Dict[str, Any]:
    """Accepts a Streamlit uploaded file (UploadedFile) and returns key structure."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    if name.endswith(".json"):
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Invalid JSON key: {e}")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            return load_key_from_excel_bytes(raw)
        except Exception as e:
            raise RuntimeError(f"Invalid Excel key: {e}")
    raise RuntimeError("Unsupported key format. Provide JSON or Excel (.xlsx/.xls).")

# ---------------------------
# Helper image functions
# ---------------------------
def find_document_contour(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:40]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            return approx
    return None

def correct_orientation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 250))
    if coords.shape[0] == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h,w) = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# ---------------------------
# Adaptive contour-based bubble mode (PyImageSearch style)
# ---------------------------
def grade_contour_mode(img_bgr: np.ndarray, answer_key: Dict[str,Any],
                       absolute_min: int = 30, ambiguous_ratio: float = 0.8, debug: bool=False):
    """
    img_bgr: original BGR image
    answer_key: dict with 'answers' list and 'num_choices' and 'num_questions'
    Returns: results_dict, annotated_image
    """
    img = img_bgr.copy()
    # deskew small preview for doc detection
    small = imutils.resize(img, height=700)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    docCnt = find_document_contour(edged)

    # warp full-resolution image using scaled doc contour
    scale = img_bgr.shape[0] / float(small.shape[0])
    if docCnt is None:
        warped_color = img_bgr.copy()
        warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    else:
        docCnt_scaled = (docCnt.reshape(4,2) * scale).astype("float32")
        warped_color = four_point_transform(img_bgr, docCnt_scaled)
        warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)

    # threshold & find contours
    thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # filter bubble-like contours by size & aspect ratio
    questionCnts = []
    for c in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        ar = w / float(h) if h>0 else 0
        # heuristics: bubble approx square-ish and not tiny
        if w >= 15 and h >= 15 and 0.6 <= ar <= 1.6:
            questionCnts.append(c)

    # if debug, collect images
    debug_imgs = {}
    if debug:
        debug_imgs['edged'] = edged
        debug_imgs['warped_gray'] = warped_gray
        debug_imgs['thresh'] = thresh
        tmp = warped_color.copy()
        cv2.drawContours(tmp, questionCnts, -1, (0,0,255), 2)
        debug_imgs['bubble_candidates'] = tmp

    if len(questionCnts) == 0:
        raise RuntimeError("No bubble-like contours found in adaptive mode")

    # sort top-to-bottom then group into rows of num_choices
    questionCnts = imcontours.sort_contours(questionCnts, method="top-to-bottom")[0]

    num_choices = int(answer_key.get("num_choices", 4))
    num_questions = int(answer_key.get("num_questions", len(answer_key.get("answers", []))))

    # Now build groups: we expect num_questions * num_choices bubbles ideally.
    # But contours detection may find extras. We'll group by y-coordinate into rows then select best num_choices per row.
    # compute centers and y's
    centers = []
    bboxes = []
    for c in questionCnts:
        x,y,w,h = cv2.boundingRect(c)
        centers.append((x + w//2, y + h//2))
        bboxes.append((x,y,w,h,c))
    ys = [pt[1] for pt in centers]
    if len(ys) == 0:
        raise RuntimeError("No bubble centers found")

    # Use kmeans-like binning by splitting y-range into num_questions rows
    miny, maxy = min(ys), max(ys)
    if num_questions <= 0:
        raise RuntimeError("Invalid number of questions in key")
    bins = np.linspace(miny-1, maxy+1, num_questions+1)
    rows = [[] for _ in range(num_questions)]
    for bbox, c, cy in zip(bboxes, centers, ys):
        idx = np.searchsorted(bins, cy) - 1
        idx = max(0, min(num_questions-1, idx))
        rows[idx].append(bbox)  # append (x,y,w,h,c)

    # For safety, if some rows empty, try to redistribute by approximate equal counts
    # If number of detected rows with nonzero length not equal to num_questions, we may fallback to ROI/grid later.
    grouped = []
    for row in rows:
        if len(row) == 0:
            grouped.append([])
            continue
        # sort row left-to-right by x
        row_sorted = sorted(row, key=lambda r: r[0])
        # if more than num_choices, take largest num_choices by area (likely real)
        if len(row_sorted) > num_choices:
            # sort by x to try to keep left-most choices; but if bubble detection found extras, choose by x
            row_sorted = row_sorted[:num_choices]
        grouped.append([r[4] for r in row_sorted])  # append contour only

    # If grouped rows have many empties or inconsistent counts, we'll consider a fallback to ROI-grid.
    row_counts = [len(r) for r in grouped]
    too_sparse = sum(1 for c in row_counts if c < max(1, num_choices//2)) > (0.3 * num_questions)

    final_results = []
    annotated = warped_color.copy()

    if (not too_sparse):
        # process each question row
        for q_idx, row_contours in enumerate(grouped):
            if len(row_contours) == 0:
                final_results.append({
                    "question": q_idx+1,
                    "detected": None,
                    "counts": [],
                    "confidences": [],
                    "status": "no_bubbles_found"
                })
                continue
            # ensure we have num_choices (if fewer, still proceed)
            # sort row left-to-right
            row_sorted = imcontours.sort_contours(row_contours, method="left-to-right")[0]
            counts = []
            boxes = []
            for c in row_sorted:
                (x,y,w,h) = cv2.boundingRect(c)
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                masked = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = int(cv2.countNonZero(masked))
                counts.append(total)
                boxes.append((x,y,w,h))
            # compute detection
            counts_arr = np.array(counts)
            if counts_arr.size == 0:
                final_results.append({
                    "question": q_idx+1,
                    "detected": None, "counts": [], "confidences": [], "status": "no_bubbles_found"
                })
                continue
            max_val = int(np.max(counts_arr))
            if max_val < absolute_min:
                # everything under threshold -> no_mark
                confidences = [round(float(c)/ (counts_arr.sum() + 1e-9) * 100, 2) for c in counts_arr]
                final_results.append({
                    "question": q_idx+1, "detected": None, "counts": counts, "confidences": confidences, "status": "no_mark"
                })
                continue
            # find candidates >= threshold (e.g., >= 50% of max)
            threshold = max(absolute_min, int(max_val * 0.5))
            marked = [i for i, val in enumerate(counts_arr) if val >= threshold]
            confidences = [round(float(c)/ (counts_arr.sum() + 1e-9) * 100, 2) for c in counts_arr]
            if len(marked) == 0:
                final_results.append({
                    "question": q_idx+1, "detected": None, "counts": counts, "confidences": confidences, "status": "no_mark"
                })
                continue
            if len(marked) == 1:
                detected = marked[0]
                # check ambiguity: second best >= ambiguous_ratio * best
                sorted_vals = sorted(counts_arr, reverse=True)
                ambiguous = False
                if len(sorted_vals) >= 2 and sorted_vals[1] >= sorted_vals[0] * ambiguous_ratio:
                    ambiguous = True
                correct_choice = answer_key["answers"][q_idx] if q_idx < len(answer_key["answers"]) else None
                status = "unknown_key" if correct_choice is None else ("correct" if detected == correct_choice else "incorrect")
                if ambiguous:
                    status += "_ambiguous"
                # annotate chosen
                if detected < len(boxes):
                    (x,y,w,h) = boxes[detected]
                    color = (0,255,0) if status.startswith("correct") else (0,0,255)
                    cv2.rectangle(annotated, (x,y), (x+w, y+h), color, 2)
                final_results.append({
                    "question": q_idx+1, "detected": detected, "counts": counts, "confidences": confidences, "status": status
                })
            else:
                # multiple marks
                detected = marked
                # annotate all marked
                for idx in detected:
                    if idx < len(boxes):
                        x,y,w,h = boxes[idx]
                        cv2.rectangle(annotated, (x,y), (x+w, y+h), (0,200,200), 2)
                final_results.append({
                    "question": q_idx+1, "detected": detected, "counts": counts, "confidences": confidences, "status": "multiple_marks"
                })
    else:
        # too sparse detection -> fallback to ROI/grid scanning (if ROI provided)
        if "roi" in answer_key:
            # call ROI-grid routine (reuse simple grid approach)
            grid_res, grid_ann = grade_roi_grid(warped_color, answer_key, absolute_min=absolute_min)
            # paste grid_ann onto annotated
            annotated = grid_ann
            final_results = grid_res["results"]
        else:
            # fallback: attempt automatic block detection + grid
            try:
                grid_res, grid_ann = grade_large_box_then_grid(warped_color, answer_key, absolute_min=absolute_min)
                annotated = grid_ann
                final_results = grid_res["results"]
            except Exception as e:
                # cannot fallback, mark all as error
                raise RuntimeError("Contour-based detection too sparse and ROI/grid fallback failed: " + str(e))

    # summary
    correct = sum(1 for r in final_results if r.get("status") and str(r.get("status")).startswith("correct"))
    wrong = sum(1 for r in final_results if r.get("status") == "incorrect")
    no_mark = sum(1 for r in final_results if r.get("status") == "no_mark")
    multiple = sum(1 for r in final_results if r.get("status") in ("multiple_marks", "ambiguous", "ambiguous_marks"))
    total = int(answer_key.get("num_questions", len(answer_key.get("answers", []))))
    score = (correct/total)*100.0 if total>0 else 0.0

    results_dict = {
        "results": final_results,
        "correct": int(correct),
        "wrong": int(wrong),
        "no_mark": int(no_mark),
        "multiple": int(multiple),
        "total_questions": int(total),
        "score_percent": round(score,2),
        "debug_images": debug_imgs if debug else {}
    }
    return results_dict, annotated

# ---------------------------
# ROI/grid grader (fallback / template)
# ---------------------------
def grade_roi_grid(warped_color: np.ndarray, answer_key: Dict[str,Any], absolute_min:int=30):
    """
    Simple fixed-grid grading inside given ROI in answer_key.
    answer_key must contain 'roi' (y1,y2,x1,x2) in pixels relative to warped_color.
    """
    if "roi" not in answer_key:
        raise RuntimeError("ROI not present in key for grid grading")
    y1,y2,x1,x2 = map(int, answer_key["roi"][:4])
    h, w = warped_color.shape[:2]
    y1,y2 = max(0,y1), min(h,y2)
    x1,x2 = max(0,x1), min(w,x2)
    crop = warped_color[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    num_q = int(answer_key["num_questions"])
    num_choices = int(answer_key.get("num_choices",4))
    ch, cw = gray.shape
    cell_h = ch // num_q
    cell_w = cw // num_choices
    annotated_crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    results=[]
    for q in range(num_q):
        counts=[]
        for c in range(num_choices):
            yA = q*cell_h; yB = min(yA+cell_h, ch)
            xA = c*cell_w; xB = min(xA+cell_w, cw)
            roi = gray[yA:yB, xA:xB]
            _, thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            cnt = int(cv2.countNonZero(thr))
            counts.append(cnt)
        maxval = max(counts) if counts else 0
        threshold = max(absolute_min, int(maxval*0.5))
        marked = [i for i,val in enumerate(counts) if val>=threshold]
        confidences = [round((c/(sum(counts)+1e-9))*100,2) for c in counts]
        if len(marked)==0:
            status="no_mark"; detected=None
        elif len(marked)==1:
            detected=marked[0]; correct_choice = answer_key["answers"][q] if q<len(answer_key["answers"]) else None
            status = "unknown_key" if correct_choice is None else ("correct" if detected==correct_choice else "incorrect")
        else:
            detected=marked; status="multiple_marks"
        results.append({"question": q+1, "counts":counts, "confidences":confidences, "detected":detected, "correct_choice": answer_key["answers"][q] if q<len(answer_key["answers"]) else None, "status":status})
        # annotate each cell
        for c in range(num_choices):
            xA = c*cell_w; yA = q*cell_h; xB=min(xA+cell_w,cw); yB=min(yA+cell_h,ch)
            color=(180,180,180); thickness=1
            if isinstance(detected,int) and detected==c:
                color=(0,255,0) if status.startswith("correct") else (0,0,255)
                thickness=2
            if isinstance(detected,list) and c in detected:
                color=(0,200,200); thickness=2
            annotated_crop = cv2.rectangle(annotated_crop, (xA,yA), (xB,yB), color, thickness)
    # paste annotated crop back into warped_color image
    full_ann = warped_color.copy()
    ch2, cw2 = annotated_crop.shape[:2]
    full_ann[y1:y1+ch2, x1:x1+cw2] = annotated_crop
    results_dict = {
        "results": results,
        "correct": sum(1 for r in results if r.get("status") and str(r.get("status")).startswith("correct")),
        "wrong": sum(1 for r in results if r.get("status")=="incorrect"),
        "no_mark": sum(1 for r in results if r.get("status")=="no_mark"),
        "multiple": sum(1 for r in results if r.get("status")=="multiple_marks"),
        "total_questions": num_q,
        "score_percent": round( (sum(1 for r in results if r.get("status") and str(r.get("status")).startswith("correct")) / max(1,num_q)) * 100.0, 2)
    }
    return results_dict, full_ann

# ---------------------------
# Large-box-first: find largest rectangular box and do grid inside it
# ---------------------------
def find_largest_rectangular_contour(gray_img) -> Optional[np.ndarray]:
    _, th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnts = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            return approx
    return None

def grade_large_box_then_grid(warped_color, answer_key, absolute_min=30):
    gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    big = find_largest_rectangular_contour(gray)
    if big is None:
        raise RuntimeError("No large rectangular box found for large-box scan")
    # warp that box to a normalized rectangle then call ROI grid using that crop as whole sheet
    block = four_point_transform(warped_color, big.reshape(4,2))
    # now grade block as if it's full sheet with ROI covering whole block
    # create synthetic key with roi covering full block and same numbers
    synthetic_key = answer_key.copy()
    synthetic_key = synthetic_key.copy()
    synthetic_key['roi'] = [0, block.shape[0], 0, block.shape[1]]
    # call grade_roi_grid with block used as warped_color
    results_dict, full_ann = grade_roi_grid(block, synthetic_key, absolute_min=absolute_min)
    # full_ann corresponds to block; we can return it (caller will place into full warped image if needed)
    return results_dict, full_ann

# ---------------------------
# Streamlit UI / batch handling
# ---------------------------
st.title("OMR Grader — Adaptive (contour) + ROI/grid fallback")
st.write("Upload answer key (Excel or JSON) and one or more sheet images (.jpg/.png/.tiff) or PDFs. "
         "Choose adaptive contour mode (auto bubble detection) or force ROI/grid if your template is fixed.")

uploaded_files = st.file_uploader("Upload sheet images (or PDFs)", accept_multiple_files=True,
                                  type=['jpg','jpeg','png','tif','tiff','pdf'])
key_file = st.file_uploader("Upload Answer Key (.xlsx/.xls or .json)", type=['xlsx','xls','json'])
poppler_path = st.text_input("Poppler bin path (leave empty if poppler on PATH)", value="")

st.sidebar.header("Options")
mode = st.sidebar.selectbox("Scanning mode", ["adaptive_contour", "roi_grid", "large_box_then_grid"])
absolute_min = st.sidebar.number_input("Absolute min pixels threshold", value=30, min_value=1)
ambiguous_ratio = st.sidebar.slider("Ambiguity second-best / best ratio", 0.5, 0.95, 0.8)
debug_images = st.sidebar.checkbox("Show debug (intermediate) images", value=False)
num_workers = st.sidebar.slider("Parallel workers", 1, 8, 4)

key_struct = None
set_choice = None
subject_choice = None
if key_file:
    try:
        key_struct = load_answer_key_file(key_file)
        st.success("Answer key loaded")
    except Exception as e:
        st.error(f"Failed to load key: {e}")
        key_struct = None

    # if workbook style
    if isinstance(key_struct, dict) and any(isinstance(v, dict) and "answers" not in v for v in key_struct.values()):
        st.info("Detected multi-sheet key. Choose which set/sheet to use.")
        set_choice = st.selectbox("Choose set", list(key_struct.keys()))
        branch = key_struct.get(set_choice, {})
        # if branch contains subjects
        if isinstance(branch, dict) and any(isinstance(v, dict) and "answers" in v for v in branch.values()):
            subject_choice = st.selectbox("Choose subject", list(branch.keys()))

if st.button("Start grading"):
    if not uploaded_files or not key_file:
        st.error("Upload at least one sheet and a key")
        st.stop()

    # resolve key
    try:
        if isinstance(key_struct, dict) and "answers" in key_struct:
            answer_key = key_struct
            used_set, used_subject = None, None
        else:
            if set_choice is None:
                set_choice = list(key_struct.keys())[0]
            branch = key_struct[set_choice]
            if isinstance(branch, dict) and "answers" in branch:
                answer_key = branch
                used_set, used_subject = set_choice, None
            else:
                if subject_choice is None:
                    subject_choice = list(branch.keys())[0]
                answer_key = branch[subject_choice]
                used_set, used_subject = set_choice, subject_choice
    except Exception as e:
        st.error(f"Failed to resolve key: {e}")
        st.stop()

    # sanitize
    if "answers" not in answer_key:
        st.error("Answer key missing 'answers' list")
        st.stop()
    answer_key["answers"] = [int(a) if a is not None else None for a in answer_key["answers"]]
    if "num_choices" not in answer_key:
        answer_key["num_choices"] = 4
    if "num_questions" not in answer_key:
        answer_key["num_questions"] = len(answer_key["answers"])

    # prepare image tasks (expand PDFs)
    tasks = []
    for up in uploaded_files:
        fname = up.name
        lower = fname.lower()
        if lower.endswith(".pdf"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tpdf:
                    tpdf.write(up.read()); tpdf.flush(); tmp_pdf = tpdf.name
                pages = convert_from_path(tmp_pdf, dpi=200, poppler_path=poppler_path.strip() or None)
                os.unlink(tmp_pdf)
                for i, page in enumerate(pages):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpimg:
                        page.save(tmpimg.name, "JPEG")
                        with open(tmpimg.name,"rb") as f:
                            b = f.read()
                        basename = f"{os.path.splitext(fname)[0]}_page{i+1}"
                        tasks.append((b, basename))
                        os.unlink(tmpimg.name)
            except Exception as e:
                st.error(f"PDF conversion failed for {fname}: {e}")
        else:
            b = up.read(); basename = os.path.splitext(fname)[0]; tasks.append((b, basename))

    st.info(f"Prepared {len(tasks)} images for grading")
    tmp_root = tempfile.mkdtemp(prefix="omr_batch_")
    out_dir = os.path.join(tmp_root, "outputs"); os.makedirs(out_dir, exist_ok=True)

    results_summary = []
    progress = st.progress(0)
    status = st.empty()

    def process_task(task):
        image_bytes, basename = task
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Unable to decode image")
            if mode == "adaptive_contour":
                res, ann = grade_contour_mode(img, answer_key, absolute_min=absolute_min, ambiguous_ratio=ambiguous_ratio, debug=debug_images)
            elif mode == "roi_grid":
                # warp then grade grid (use doc contour)
                small = imutils.resize(img, height=700); gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                docCnt = find_document_contour(cv2.Canny(cv2.GaussianBlur(gray, (5,5),0), 75,200))
                if docCnt is not None:
                    scale = img.shape[0] / float(small.shape[0])
                    docCnt_scaled = (docCnt.reshape(4,2) * scale).astype("float32")
                    warped = four_point_transform(img, docCnt_scaled)
                else:
                    warped = img.copy()
                res, ann = grade_roi_grid(warped, answer_key, absolute_min=absolute_min)
            elif mode == "large_box_then_grid":
                small = imutils.resize(img, height=700); gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                docCnt = find_document_contour(cv2.Canny(cv2.GaussianBlur(gray, (5,5),0), 75,200))
                if docCnt is not None:
                    scale = img.shape[0] / float(small.shape[0])
                    docCnt_scaled = (docCnt.reshape(4,2) * scale).astype("float32")
                    warped = four_point_transform(img, docCnt_scaled)
                else:
                    warped = img.copy()
                res, ann = grade_large_box_then_grid(warped, answer_key, absolute_min=absolute_min)
            else:
                raise RuntimeError("Unknown mode")
            # save annotated and csv
            out_img = os.path.join(out_dir, f"{basename}_annotated.png")
            cv2.imwrite(out_img, ann)
            rows=[]
            for q in res["results"]:
                rows.append({
                    "question": q.get("question"),
                    "detected": q.get("detected"),
                    "correct_choice": q.get("correct_choice"),
                    "status": q.get("status"),
                    "counts": "|".join(map(str,q.get("counts",[]))),
                    "confidences": "|".join(map(str, q.get("confidences",[])))
                })
            df = pd.DataFrame(rows)
            summary_row = {
                "question":"Score",
                "detected":"",
                "correct_choice": f"{res.get('correct')}/{res.get('total_questions')}",
                "status": f"{res.get('score_percent'):.2f}%"
            }
            df.loc[len(df)] = summary_row
            out_csv = os.path.join(out_dir, f"{basename}_results.csv")
            df.to_csv(out_csv, index=False)
            return {"base": basename, "img": out_img, "csv": out_csv, "score": res.get("score_percent"), "correct": res.get("correct"), "wrong": res.get("wrong"), "no_mark": res.get("no_mark"), "multiple": res.get("multiple")}
        except Exception as e:
            return {"base": basename, "error": str(e)}

    # parallel execution
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(process_task, t) for t in tasks]
        total = len(futures); completed=0
        for fut in as_completed(futures):
            info = fut.result()
            results_summary.append(info)
            completed += 1
            progress.progress(int(100*completed/total))
            status.text(f"Processed {completed}/{total}")

    # aggregate table
    agg=[]
    for r in results_summary:
        if "error" in r:
            agg.append({"sheet": r.get("base"), "score": None, "correct": None, "wrong": None, "no_mark": None, "multiple": None, "issues": r.get("error")})
        else:
            issues=[]
            if r.get("no_mark"): issues.append(f"no_mark:{r['no_mark']}")
            if r.get("multiple"): issues.append(f"multiple:{r['multiple']}")
            agg.append({"sheet": r["base"], "score": r["score"], "correct": r["correct"], "wrong": r["wrong"], "no_mark": r["no_mark"], "multiple": r["multiple"], "issues": ", ".join(issues)})
    agg_df = pd.DataFrame(agg)
    agg_csv = os.path.join(out_dir, "aggregate_results.csv")
    agg_df.to_csv(agg_csv, index=False)

    # make zip
    buff = BytesIO()
    with zipfile.ZipFile(buff, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results_summary:
            if "img" in r and os.path.exists(r["img"]):
                zf.write(r["img"], arcname=os.path.basename(r["img"]))
            if "csv" in r and os.path.exists(r["csv"]):
                zf.write(r["csv"], arcname=os.path.basename(r["csv"]))
        zf.write(agg_csv, arcname=os.path.basename(agg_csv))
    buff.seek(0)

    st.success("Batch grading finished")
    st.write("## Aggregate summary")
    st.dataframe(agg_df)
    st.download_button("Download results ZIP", buff.getvalue(), file_name="omr_results.zip", mime="application/zip")

    st.write("## Annotated preview (first 10)")
    for r in results_summary[:10]:
        if "img" in r and os.path.exists(r["img"]):
            st.image(r["img"], caption=f"{r['base']} — {r.get('score',0):.2f}%", use_column_width=False)
            with open(r["img"], "rb") as f:
                st.download_button(f"Download {os.path.basename(r['img'])}", f.read(), file_name=os.path.basename(r['img']))

    if st.button("Clear temporary outputs"):
        try:
            shutil.rmtree(tmp_root)
            st.success("Temporary outputs removed")
        except Exception as e:
            st.error(f"Cleanup failed: {e}")
