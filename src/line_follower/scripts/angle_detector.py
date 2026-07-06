"""
angle_detector.py

Given an image containing a single dark line (crack/tape/path) on a noisy
gray background, this figures out:
  1. Which blob of "dark pixels" is actually the line the person should
     follow (the one touching the BOTTOM edge of the frame — i.e. the
     line that reaches the floor the person is standing on).
  2. The farthest point of that line from the person (bottom-center of
     the image) — this is the "distance point" to steer toward.
  3. The turn angle (in degrees, 0 = straight ahead, +right / -left)
     needed to face that point.

Works even if:
  - The line is broken/faded in places (morphological closing bridges gaps).
  - There are OTHER dark blobs/lines in the frame that don't touch the
    bottom edge (they get filtered out — only bottom-connected blobs
    are considered candidates).
  - The image is heavy with grain/sensor noise (median blur + Otsu
    threshold handle that).
"""

import cv2
import numpy as np
import math
import os


def find_line_angle(image_path, debug_dir=None, bottom_touch_tolerance=4):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    bottom_center = (w // 2, h - 1)

    # --- 1. Denoise ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    # --- 2. Threshold: the line is much darker than the noisy gray bg ---
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --- 3. Morphological close: bridge small gaps/broken segments ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # remove tiny speckle noise
    closed = cv2.morphologyEx(
        closed, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    # --- 4. Connected components ---
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    # --- 5. Keep only components that touch the BOTTOM edge (the floor) ---
    candidates = []
    for i in range(1, num):  # skip background label 0
        y0, y1 = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if y1 >= h - bottom_touch_tolerance and area > 20:
            candidates.append(i)

    if not candidates:
        return {"found": False, "reason": "no dark blob touches the bottom edge"}

    # pick the largest bottom-touching blob = the line we follow
    best = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    ys, xs = np.where(labels == best)

    # --- 6. Fit the ROAD'S CENTER AXIS (not just grab an extreme pixel) ---
    # A raw "farthest pixel" is biased toward whichever edge of a thick
    # stripe happens to poke out furthest. Instead, fit a straight line
    # through ALL the blob's pixels (this averages across the road's
    # width, landing on its centerline) and walk along that line from
    # the bottom until it exits the frame.
    pts = np.column_stack((xs, ys)).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    # Make sure the direction vector points AWAY from the person (i.e.
    # toward whichever raw extreme pixel is farthest from bottom_center)
    d2 = (xs - bottom_center[0]) ** 2 + (ys - bottom_center[1]) ** 2
    ref_idx = int(np.argmax(d2))
    ref_pt = np.array([xs[ref_idx], ys[ref_idx]], dtype=np.float32)
    if (ref_pt[0] - x0) * vx + (ref_pt[1] - y0) * vy < 0:
        vx, vy = -vx, -vy

    # Walk from the fitted line's own point along (vx, vy) and find where
    # it exits the image (top / left / right edge — never bottom, since
    # we're heading away from the person).
    candidates_t = []
    if vy != 0:
        t = (0 - y0) / vy  # top edge, y = 0
        x_at = x0 + t * vx
        if t > 0 and 0 <= x_at <= w - 1:
            candidates_t.append(t)
    if vx != 0:
        t = (0 - x0) / vx  # left edge, x = 0
        y_at = y0 + t * vy
        if t > 0 and 0 <= y_at <= h - 1:
            candidates_t.append(t)
        t = (w - 1 - x0) / vx  # right edge, x = w-1
        y_at = y0 + t * vy
        if t > 0 and 0 <= y_at <= h - 1:
            candidates_t.append(t)

    if candidates_t:
        t_best = min(candidates_t)  # first boundary the line hits
        far_pt = (int(round(x0 + t_best * vx)), int(round(y0 + t_best * vy)))
    else:
        # fallback: shouldn't normally happen, use the raw extreme pixel
        far_pt = (int(ref_pt[0]), int(ref_pt[1]))

    # --- 7. Turn angle: 0 = straight up/forward, + = turn right, - = turn left
    dx = far_pt[0] - bottom_center[0]
    dy = bottom_center[1] - far_pt[1]  # image y grows downward
    angle_deg = math.degrees(math.atan2(dx, dy))

    result = {
        "found": True,
        "bottom_center": bottom_center,
        "distance_point": far_pt,
        "angle_deg": round(angle_deg, 2),
        "blob_area": int(stats[best, cv2.CC_STAT_AREA]),
    }

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        vis = img.copy()
        cv2.line(vis, bottom_center, far_pt, (0, 0, 255), 2)
        cv2.circle(vis, far_pt, 6, (0, 255, 0), -1)
        cv2.circle(vis, bottom_center, 6, (255, 0, 0), -1)
        cv2.putText(vis, f"{angle_deg:.1f} deg", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        name = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(debug_dir, f"{name}_mask.png"), closed)
        cv2.imwrite(os.path.join(debug_dir, f"{name}_result.png"), vis)

    return result


def process_one(image_path, output_dir, log=print):
    """Run detection on a single image, draw the result, save it into
    output_dir renamed to its angle. Returns the result dict."""
    result = find_line_angle(image_path)

    if not result["found"]:
        log(f"[SKIP] {image_path}: {result['reason']}")
        return result

    img = cv2.imread(image_path)
    bottom_center = result["bottom_center"]
    far_pt = result["distance_point"]
    angle = result["angle_deg"]

    cv2.line(img, bottom_center, far_pt, (0, 0, 255), 2)
    cv2.circle(img, far_pt, 6, (0, 255, 0), -1)
    cv2.circle(img, bottom_center, 6, (255, 0, 0), -1)
    cv2.putText(img, f"{angle:.2f} deg", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    ext = os.path.splitext(image_path)[1] or ".png"
    base_name = f"{angle:.2f}"
    out_path = os.path.join(output_dir, base_name + ext)
    # avoid overwriting if two images land on the same angle
    counter = 1
    while os.path.exists(out_path):
        out_path = os.path.join(output_dir, f"{base_name}_{counter}{ext}")
        counter += 1

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(out_path, img)
    log(f"[OK] {image_path} -> {out_path}  ({angle:.2f} deg)")
    result["output_path"] = out_path
    return result


def process_path(input_path, output_dir, log=print):
    """input_path can be a single image file or a folder of images."""
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    if os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in valid_ext
        )
        if not files:
            log(f"No image files found in folder: {input_path}")
            return []
        return [process_one(f, output_dir, log=log) for f in files]
    elif os.path.isfile(input_path):
        return [process_one(input_path, output_dir, log=log)]
    else:
        log(f"Path not found: {input_path}")
        return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect the line angle in an image (or a whole folder "
                    "of images), draw the result, and save it renamed to "
                    "its angle in a separate output folder."
    )
    parser.add_argument("input", help="Path to a single image OR a folder of images")
    parser.add_argument(
        "-o", "--output", default="angled_output",
        help="Folder to save the renamed, annotated results into "
             "(default: ./angled_output)"
    )
    args = parser.parse_args()

    process_path(args.input, args.output)