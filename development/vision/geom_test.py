
import cv2
import numpy as np
from scipy.optimize import least_squares

from dataclasses import dataclass
import subprocess
# import sys
import time
# from pathlib import Path

# import cv2
# import numpy as np

# if __package__ in (None, ""):
#     sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# import config





# ============================================================
# GET IMAGE
# ============================================================
# image = cv2.imread("image.jpg")  # if we want to use a test image
capture_args = ['libcamera-jpeg', '-o', '-', '--width', '3280', '--height', '2464', '--nopreview']
result = subprocess.run(capture_args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
image = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # immediately convert to grayscale to speed up processing

# ============================================================
# DISTORT
# ============================================================
original_camera_matrix = np.array([
    [2650.9964176101485, 0.0, 1638.4541745042789],
    [0.0, 2644.434621507731, 1106.6683119862594],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

distortion_coefficients = np.array([
    0.19930490026457645,
    -0.2903695599399088,
    -0.007406225233953261,
    0.005002318775684634,
    -0.5143858238250185
], dtype=np.float32)

camera_height = 112.1 # mm
camera_pitch_degrees = 47.2 # degrees

image_height, image_width = image.shape[:2]
image_size = (image_width, image_height)

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(original_camera_matrix, distortion_coefficients, image_size, alpha=1)
undistorted_image = cv2.undistort(image, original_camera_matrix, distortion_coefficients, None, new_camera_matrix)

# ============================================================
# PRECPROCESS
# ============================================================
blurred = cv2.GaussianBlur(gray, (9, 9), 0) # update to use the undistorted image eventually
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
binary = cv2.bitwise_not(binary)  # lines/ink become white
kernel = np.ones(3 * 2, np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# ============================================================
# GET CORNERS
# ============================================================
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
for contour in contours:
    approximation = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approximation) == 4:
        break
else:
    raise ValueError("Could not find Sudoku board")

# Sort TL, TR, BR, BL
points = approximation.reshape(4, 2).astype(np.float32)                 # Get corner points from contours
points_sorted_by_y = points[np.argsort(points[:, 1])]                   # Sort by y-coordinate to separate top and bottom rows
top_row = points_sorted_by_y[:2]                                        # Extract top row
bottom_row = points_sorted_by_y[2:]                                     # Extract bottom row
top_left, top_right = top_row[np.argsort(top_row[:, 0])]
bottom_left, bottom_right = bottom_row[np.argsort(bottom_row[:, 0])]
corner_pixels = np.array([top_left, top_right, bottom_right, bottom_left],dtype=np.float32,)


# ============================================================
# 3. CONVERT PIXEL COORDINATES INTO CAMERA RAYS
# ============================================================
inverse_camera_matrix = np.linalg.inv(original_camera_matrix)
homogeneous_pixels = np.column_stack((corner_pixels, np.ones(4))) # [u, v] --> [u, v, 1].
camera_rays = (inverse_camera_matrix @ homogeneous_pixels.T).T # in camera frame assuming it faces forward
camera_pitch = np.radians(camera_pitch_degrees)
cos_pitch = np.cos(camera_pitch)
sin_pitch = np.sin(camera_pitch)
pitch_rotation_matrix = np.array([
    [1.0,        0.0,         0.0],
    [0.0, cos_pitch, -sin_pitch],
    [0.0, sin_pitch,  cos_pitch]
], dtype=np.float64)
ground_rays = (pitch_rotation_matrix @ camera_rays.T).T


# ============================================================
# 5. INTERSECT EACH RAY WITH THE GROUND PLANE
# ============================================================

# P(t) = camera_position + t * ground_ray
# t = -camera_height / ray_z
camera_position = np.array([0.0, 0.0, camera_height])
ray_parameters = (-camera_height / ground_rays[:, 2])
ground_points_3d = (camera_position + ray_parameters[:, np.newaxis] * ground_rays)
ground_points = ground_points_3d[:, :2] # discard Z
bottom_left_ground_point = ground_points[3].copy()
ground_points -= bottom_left_ground_point


# ============================================================
# OUTPUTS
# ============================================================
print("\nCorner geometry")
print("==============")
for index, (pixel, camera_ray, ground_ray, ray_parameter, ground_point) in enumerate(
    zip(corner_pixels, camera_rays, ground_rays, ray_parameters, ground_points),
    start=1,
):
    corner_name = ("top-left", "top-right", "bottom-right", "bottom-left")[index - 1]
    print(f"{corner_name}:")
    print(f"  pixel coordinates: ({pixel[0]:.2f}, {pixel[1]:.2f})")
    print(f"  camera-frame ray:  {np.array2string(camera_ray, precision=6)}")
    print(f"  global-frame ray:  {np.array2string(ground_ray, precision=6)}")
    print(f"  t (Z=0):           {ray_parameter:.6f}")
    print(f"  ground point (mm): {np.array2string(ground_point, precision=6)}")



# ============================================================
# 6. FIT AN IDEAL RECTANGLE TO THE FOUR GROUND POINTS
# ============================================================

# def calculate_rectangle_corners(parameters):
#     centre_x, centre_y, width, height, angle = parameters
#     centre = np.array([centre_x, centre_y])
#     side_a = np.array([np.cos(angle), np.sin(angle)])
#     side_b = np.array([-np.sin(angle), np.cos(angle)])
#     corner_1 = centre - (width / 2) * side_a - (height / 2) * side_b
#     corner_2 = centre + (width / 2) * side_a - (height / 2) * side_b
#     corner_3 = centre + (width / 2) * side_a + (height / 2) * side_b
#     corner_4 = centre - (width / 2) * side_a + (height / 2) * side_b
#     return np.array([corner_1, corner_2, corner_3, corner_4])

# def rectangle_residuals(parameters):
#     estimated_corners = calculate_rectangle_corners(parameters)
#     return (ground_points - estimated_corners).ravel()

# initial_centre = np.mean(ground_points,axis=0)                          # Estimate the rectangle centre as the average of the four observed ground-plane points.
# initial_width = np.linalg.norm(ground_points[1] - ground_points[0])     # Estimate width from the distance between corners 0 and 1.
# initial_height = np.linalg.norm(ground_points[2] - ground_points[1])    # Estimate height from the distance between corners 1 and 2.
# initial_angle = np.arctan2(ground_points[1, 1] - ground_points[0, 1], ground_points[1, 0] - ground_points[0, 0]) # Estimate the initial orientation from the first side.
# initial_parameters = np.array([initial_centre[0],initial_centre[1],initial_width,initial_height,initial_angle]) # Combine the initial estimates into one parameter array.

# optimisation_result = least_squares(
#     rectangle_residuals,
#     initial_parameters,
#     bounds=([-np.inf, -np.inf, 1e-6, 1e-6, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]))

# (fitted_centre_x,fitted_centre_y,fitted_width,fitted_height,fitted_angle) = optimisation_result.x # Extract the optimised rectangle parameters.
# fitted_corners = calculate_rectangle_corners(optimisation_result.x)  # Calculate the four ideal fitted rectangle corners.
# fitted_angle_degrees = np.degrees(fitted_angle)  # Convert the orientation from radians to degrees.
# corner_errors = ground_points - fitted_corners  # Calculate the residual error between observed and fitted corners.
# total_squared_error = np.sum(corner_errors ** 2)  # Calculate the total squared corner error.
# root_mean_squared_error = np.sqrt(total_squared_error / 4)  # Calculate the corner root mean squared error.



