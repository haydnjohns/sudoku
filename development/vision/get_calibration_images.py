import subprocess
import argparse
import os
from pathlib import Path

import numpy as np
import cv2

def capture_image():
    capture_args = ["rpicam-jpeg", "-o", "-", "--width", "3280", "--height", "2464", "--nopreview"]
    process = subprocess.Popen(capture_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = process.communicate()
    image = cv2.imdecode(np.frombuffer(stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image

IMAGE_DIR = Path(__file__).resolve().parents[2] / "assets" / "vision" / "camera_calibration"


def next_image_path(directory):
    directory.mkdir(parents=True, exist_ok=True)
    counter = 0
    while True:
        path = directory / f"calibration_{counter:04d}.jpg"
        if not path.exists():
            return path
        counter += 1


def transfer_images(directory, remote_host):
    if not remote_host:
        print("No remote host supplied; images remain local.")
        return

    destination = f"{remote_host}:assets/vision/camera_calibration/"
    result = subprocess.run(["scp", "-r", str(directory), destination], check=False)
    if result.returncode == 0:
        print(f"Transferred calibration images to {destination}")
    else:
        print("Failed to transfer calibration images.")


def main():
    parser = argparse.ArgumentParser(description="Capture calibration images.")
    parser.add_argument(
        "remote_host",
        nargs="?",
        default=os.environ.get("CALIBRATION_REMOTE_HOST"),
        help="scp destination host, or set CALIBRATION_REMOTE_HOST",
    )
    args = parser.parse_args()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print("Press Enter to capture an image; enter e or q to exit.")
    while True:
        command = input("Capture> ").strip().lower()
        if command in {"e", "q"}:
            break
        if command:
            print("Press Enter to capture, or enter e/q to exit.")
            continue

        image = capture_image()
        if image is None:
            print("Capture failed.")
            continue

        image_path = next_image_path(IMAGE_DIR)
        if cv2.imwrite(str(image_path), image):
            print(f"Captured image: {image_path}")
        else:
            print(f"Failed to save image: {image_path}")

    transfer_images(IMAGE_DIR, args.remote_host)


if __name__ == "__main__":
    main()