#!/usr/bin/env python3
"""Capture chessboard calibration images from a Raspberry Pi camera.

Press Enter to capture, q (or e) to quit. If a remote host is given (argument
or CALIBRATION_REMOTE_HOST), the images are copied there with scp on exit.
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

from picamera2 import Picamera2

IMAGE_DIR = Path(__file__).resolve().parents[2] / "assets" / "vision" / "camera_calibration"
REMOTE_DIR = "assets/vision/camera_calibration/"
RESOLUTION = (3280, 2464)


def next_image_path(directory):
    index = len(list(directory.glob("calibration_*.jpg")))
    while (path := directory / f"calibration_{index:04d}.jpg").exists():
        index += 1
    return path


def transfer_images(directory, remote_host):
    if not remote_host:
        print("No remote host supplied; images remain local.")
        return

    images = sorted(directory.glob("calibration_*.jpg"))
    if not images:
        print("No images to transfer.")
        return

    destination = f"{remote_host}:{REMOTE_DIR}"
    result = subprocess.run(["scp", *map(str, images), destination], check=False)
    if result.returncode == 0:
        print(f"Transferred {len(images)} images to {destination}")
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

    camera = Picamera2()
    camera.configure(camera.create_still_configuration(main={"size": RESOLUTION}))
    camera.options["quality"] = 95
    camera.start()
    time.sleep(2)  # let auto exposure / white balance settle

    print("Press Enter to capture an image; enter e or q to exit.")
    try:
        while True:
            try:
                command = input("Capture> ").strip().lower()
            except EOFError:
                break
            if command in {"e", "q"}:
                break
            if command:
                print("Press Enter to capture, or enter e/q to exit.")
                continue

            path = next_image_path(IMAGE_DIR)
            try:
                camera.capture_file(str(path))
            except Exception as error:
                print(f"Capture failed: {error}")
                continue
            print(f"Captured image: {path}")
    except KeyboardInterrupt:
        print()
    finally:
        camera.stop()
        camera.close()

    transfer_images(IMAGE_DIR, args.remote_host)


if __name__ == "__main__":
    main()