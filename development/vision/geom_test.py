import cv2
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# 1. LOAD IMAGE
# ============================================================
image = cv2.imread("assets/vision/geomtest.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_height, image_width = gray.shape[:2]

# ============================================================
# 2. CAMERA PROPERTIES (manual)
# ============================================================
focal_length_mm = 3.04
sensor_width_mm = 3.68
sensor_height_mm = 2.76
pixel_size_x = sensor_width_mm / image_width     # mm per pixel
pixel_size_y = sensor_height_mm / image_height

camera_height = 112.1          # mm above the ground
tilt_from_vertical_deg = 47.2    # 0 = looking straight down, 90 = looking at horizon
tilt = np.radians(tilt_from_vertical_deg)

# ============================================================
# 3. FIND CORNERS (pixels)
# ============================================================
blurred = cv2.GaussianBlur(gray, (9, 9), 0)
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
binary = cv2.bitwise_not(binary)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        break
else:
    raise ValueError("Could not find Sudoku board")

pts = approx.reshape(4, 2).astype(np.float64)
by_y = pts[np.argsort(pts[:, 1])]
top_left, top_right = by_y[:2][np.argsort(by_y[:2, 0])]
bottom_left, bottom_right = by_y[2:][np.argsort(by_y[2:, 0])]
corner_pixels = np.array([top_left, top_right, bottom_right, bottom_left])  # TL, TR, BR, BL
names = ["top-left", "top-right", "bottom-right", "bottom-left"]

# ============================================================
# 4. CAMERA FRAME: origin at sensor centre
#    X = right, Y = down the sensor, Z = away from the sensor (out of the lens)
#    Pixel (u, v) -> mm. No flip is needed because v already increases downward.
#    Each corner becomes a vector (x, y, f) pointing from the sensor centre
#    out through that pixel.
# ============================================================
u = corner_pixels[:, 0]
v = corner_pixels[:, 1]
vectors_cam = np.column_stack((
    (u - image_width / 2) * pixel_size_x,     # X (mm)
    (v - image_height / 2) * pixel_size_y,    # Y (mm)
    np.full(4, focal_length_mm),              # Z (mm)
))

print("\n=== CAMERA FRAME VECTORS (mm) [X right, Y down, Z forward] ===")
for n, p, vec in zip(names, corner_pixels, vectors_cam):
    print(f"{n:13s} pixel=({p[0]:7.1f}, {p[1]:7.1f})  vector={vec}")

# ============================================================
# 5. CAMERA FRAME -> GLOBAL FRAME (axes only, no translation yet)
#    Global axes: x = right across the page, y = up the page (away from camera),
#                 z = up out of the page.
#    Columns below are the camera's X, Y, Z axes written in global axes:
#      camera X (right)   = ( 1,         0,          0)
#      camera Y (down)    = ( 0, -cos(t),    -sin(t))
#      camera Z (forward) = ( 0,  sin(t),    -cos(t))
#    t = tilt from vertical. Check: t=0 -> Z=(0,0,-1) straight down,
#    camera Y=(0,-1,0) so image-down points back toward the camera.
# ============================================================
R_cam_to_global = np.array([
    [1.0,  0.0,          0.0],
    [0.0, -np.cos(tilt), np.sin(tilt)],
    [0.0, -np.sin(tilt), -np.cos(tilt)],
])
vectors_global = (R_cam_to_global @ vectors_cam.T).T

print("\n=== VECTORS IN GLOBAL FRAME (mm) [x right, y up page, z up] ===")
for n, vec in zip(names, vectors_global):
    print(f"{n:13s} {vec}")

# ============================================================
# 6. SCALE EACH VECTOR TO REACH THE GROUND (global z = 0)
#    Camera (lens) position in global axes: (0, 0, camera_height).
#    Point on ray = camera + K * vector.
#    z: camera_height + K * vz = 0  ->  K = -camera_height / vz
#    vz must be negative, so K must be positive.
# ============================================================
K = -camera_height / vectors_global[:, 2]

print("\n=== K (scale along each vector so global z = 0) ===")
for n, k in zip(names, K):
    print(f"{n:13s} K = {k:.4f}")
if np.any(K <= 0):
    print("WARNING: K <= 0 for at least one corner. A ray points above the horizon or the tilt convention is wrong.")

# ============================================================
# 7. GROUND POINTS, ORIGIN DIRECTLY UNDER THE CAMERA
# ============================================================
ground_under_camera = K[:, None] * vectors_global     # x, y, z (z should be -camera_height)
ground_xy_under_camera = ground_under_camera[:, :2]   # drop z (it cancels to the ground plane)

# ============================================================
# 8. GROUND FRAME: re-centre so bottom-left = (0, 0)
#    x increases right across the page, y increases up the page.
# ============================================================
bl = ground_xy_under_camera[3].copy()
ground_points = ground_xy_under_camera - bl
camera_position = -bl     # the camera was at (0, 0) before re-centring

print("\n=== GROUND FRAME (mm) [origin = bottom-left corner, x right, y up page] ===")
for n, g in zip(names, ground_points):
    print(f"{n:13s} {g}")
print(f"Camera position in ground frame (x, y): {camera_position}")
print(f"Camera height above ground (z): {camera_height}")

# ============================================================
# 9. BOARD SIZE (from the ground-frame points)
# ============================================================
TL, TR, BR, BL = ground_points
width_top, width_bottom = np.linalg.norm(TR - TL), np.linalg.norm(BR - BL)
height_left, height_right = np.linalg.norm(TL - BL), np.linalg.norm(TR - BR)
print(f"\nBoard width:  {(width_top + width_bottom) / 2:.2f} mm (top {width_top:.2f}, bottom {width_bottom:.2f})")
print(f"Board height: {(height_left + height_right) / 2:.2f} mm (left {height_left:.2f}, right {height_right:.2f})")