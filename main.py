import cv2 as cv
import numpy as np
import math
from pathlib import Path

BOARD_PATTERN = (8, 6)
BOARD_CELLSIZE = 0.025
CALIB_FILE = "calibration_data.npz"
CAMERA_INDEX = 0
WINDOW_NAME = "Star AR"

def load_calibration(calib_file):
    data = np.load(calib_file)
    K = data["K"]
    dist = data["dist"]
    if dist.ndim == 1:
        dist = dist.reshape(-1, 1)
    return K.astype(np.float64), dist.astype(np.float64)

def create_chessboard_points(pattern, size):
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= size
    return objp

def draw_text(img, text, org, color=(0, 255, 0)):
    cv.putText(img, text, org, cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, text, org, cv.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv.LINE_AA)

def draw_star(img, rvec, tvec, K, dist, scale=0.04):
    cx, cy, cz = 2, 2, -3
    outer_radius = 1.0
    inner_radius = 0.4
    star_points = []
    for i in range(10):
        angle = i * math.pi / 5
        r = outer_radius if i % 2 == 0 else inner_radius
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        z = cz
        star_points.append([x, y, z])
    star_points = np.array(star_points, dtype=np.float32) * scale
    pts_2d, _ = cv.projectPoints(star_points, rvec, tvec, K, dist)
    pts_2d = np.int32(pts_2d.reshape(-1, 2))
    cv.fillPoly(img, [pts_2d], (0, 255, 255))
    cv.polylines(img, [pts_2d], True, (0, 200, 200), 2)
    center_3d = np.array([[cx, cy, cz]], dtype=np.float32) * scale
    center_2d, _ = cv.projectPoints(center_3d, rvec, tvec, K, dist)
    c = tuple(center_2d[0][0].astype(int))
    cv.circle(img, (c[0] - 8, c[1] - 5), 3, (0, 0, 0), -1)
    cv.circle(img, (c[0] + 8, c[1] - 5), 3, (0, 0, 0), -1)
    cv.ellipse(img, (c[0], c[1] + 5), (10, 6), 0, 0, 180, (0, 0, 0), 2)

def main():
    K, dist = load_calibration(CALIB_FILE)
    obj_points = create_chessboard_points(BOARD_PATTERN, BOARD_CELLSIZE)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cap = cv.VideoCapture(CAMERA_INDEX)
    screenshot_id = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, BOARD_PATTERN, None)

        if found:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(display, BOARD_PATTERN, corners2, found)
            success, rvec, tvec = cv.solvePnP(obj_points, corners2, K, dist)

            if success:
                draw_star(display, rvec, tvec, K, dist)
                R, _ = cv.Rodrigues(rvec)
                cam_pos = (-R.T @ tvec).flatten()
                text = f"Camera XYZ: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]"
                draw_text(display, text, (10, 30))
                draw_text(display, "Cute Star AR", (10, 60), (255, 200, 0))
            else:
                draw_text(display, "solvePnP failed", (10, 30), (0, 0, 255))
        else:
            draw_text(display, "Chessboard not detected", (10, 30), (0, 0, 255))

        draw_text(display, "ESC: quit | S: save", (10, display.shape[0] - 20), (255, 255, 0))
        cv.imshow(WINDOW_NAME, display)

        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("s"):
            Path("assets").mkdir(exist_ok=True)
            path = f"assets/demo{screenshot_id}.png"
            cv.imwrite(path, display)
            screenshot_id += 1

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()