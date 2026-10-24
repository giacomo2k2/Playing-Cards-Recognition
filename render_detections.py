"""Run the main_work.py pipeline on test scenes and save annotated detection images.

Function definitions are taken verbatim from main_work.py (which runs its whole
experiment at import time, so it can't be imported directly).
Usage: python render_detections.py <test numbers...>   (e.g. python render_detections.py 5 9)
Images are written to results/detection_<n>.jpg.
"""
import ast
import os
import re
import sys

import cv2
import numpy as np

tests = [int(t) for t in sys.argv[1:]]
os.chdir(os.path.dirname(os.path.abspath(__file__)))
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

src = open("main_work.py").read()
tree = ast.parse(src)
ns = {"cv2": cv2, "np": np, "os": os, "re": re}
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) or (
        isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "card_names" for t in node.targets)
    ):
        exec(compile(ast.Module([node], []), "main_work.py", "exec"), ns)

# Parameters as set in main_work.py
segment_threshold, kernel_segmentation, iterations_segmentation = 60, 3, 5
max_key_deck, max_key_target, edge_threshold, detector = 700, 5000, 100, "orb"
lowe_ratio1 = lowe_ratio2 = 0.7
abs_thresh = 5
confidence = 0.15
min_width, min_height = 50, 80
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)

deck = ns["compute_deck_features"]("deck", max_key_deck, edge_threshold, detector)
card_names = ns["card_names"]
img_bg = cv2.imread("test_images/Background.png")
mu_bg, sigma_inv_bg = ns["compute_background_attributes"](img_bg)


def pretty(name):
    rank, suit = name.split("_")
    rank = {"1": "Ace"}.get(rank, rank.capitalize())
    return f"{rank} of {suit.capitalize()}"


for test in tests:
    img = cv2.imread(f"test_images/CardTest{test}.jpeg")
    d_M = ns["compute_mahalanobis_distance"](img, mu_bg, sigma_inv_bg)
    img_seg = ns["segment_image"](img, d_M, segment_threshold, kernel_segmentation, iterations_segmentation)
    _, mask = cv2.threshold(img_seg, 127, 255, cv2.THRESH_BINARY)
    masked = cv2.bitwise_and(img, img, mask=mask)
    _, _, stats, _ = ns["detect_blobs"](img_seg)

    out = img.copy()
    scale = img.shape[1] / 1000
    for x, y, w, h, _ in stats[1:]:
        if w < min_width or h < min_height:
            continue
        blob = masked[y:y + h, x:x + w]
        n_key = min(int(4800 / 250000 * w * h - 100), max_key_target)
        _, desc = ns["extract_keypoints_and_descriptors"](blob, n_key, False, edge_threshold, detector)
        counts = np.zeros(len(deck), dtype=int)
        if desc is not None:
            for i, card in enumerate(deck):
                for pair in bf.knnMatch(desc, card.descriptors, k=2):
                    if len(pair) < 2:
                        continue
                    m, n = pair
                    if m.distance >= lowe_ratio1 * n.distance:
                        continue
                    rev = bf.knnMatch(card.descriptors[m.trainIdx].reshape(1, -1), desc, k=2)
                    if rev and len(rev[0]) == 2:
                        p, q = rev[0]
                        if p.distance < lowe_ratio2 * q.distance and p.trainIdx == m.queryIdx:
                            counts[i] += 1
        top = counts.max()
        found = [i for i in np.argsort(counts)[::-1] if counts[i] > abs_thresh and counts[i] / top >= confidence]
        labels = [pretty(card_names[deck[i].ID]) for i in found] or ["not a card"]
        color = (60, 200, 60) if found else (150, 150, 150)
        th = max(2, int(4 * scale))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, th)
        fs = 0.9 * scale
        line_h = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][1] + int(20 * scale)
        ty = y + int(8 * scale)
        for label in labels:
            (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            cv2.rectangle(out, (x, ty), (x + tw + 12, ty + tht + base + 10), color, -1)
            cv2.putText(out, label, (x + 6, ty + tht + 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
            ty += line_h
        print(f"test {test}: blob {w}x{h} -> {labels} {[int(counts[i]) for i in found]}")
    cv2.imwrite(os.path.join(out_dir, f"detection_{test}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 85])
