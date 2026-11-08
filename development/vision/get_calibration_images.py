import subprocess
import argparse
import os
import signal
import time
from pathlib import Path

import cv2

IMAGE_DIR = Path(__file__).resolve().parents[2] / "assets" / "vision" / "camera_calibration"
CAPTURE_PATH = IMAGE_DIR / ".calibration_capture.jpg"


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


def start_camera():
    return subprocess.Popen(
        [
            "rpicam-jpeg",
            "--signal",
            "-o", str(CAPTURE_PATH),
            "--width", "3280",
            "--height", "2464",
            "--nopreview",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def capture_image(camera_process):
    if camera_process.poll() is not None:
        raise RuntimeError("Camera process exited before capture")

    previous_mtime = CAPTURE_PATH.stat().st_mtime_ns if CAPTURE_PATH.exists() else 0
    camera_process.send_signal(signal.SIGUSR1)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if CAPTURE_PATH.exists() and CAPTURE_PATH.stat().st_mtime_ns > previous_mtime:
            image = cv2.imread(str(CAPTURE_PATH), cv2.IMREAD_COLOR)
            if image is not None:
                return image
        time.sleep(0.05)
    raise RuntimeError("Timed out waiting for camera capture")


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
    camera_process = start_camera()
    print("Press Enter to capture an image; enter e or q to exit.")
    try:
        while True:
            command = input("Capture> ").strip().lower()
            if command in {"e", "q"}:
                break
            if command:
                print("Press Enter to capture, or enter e/q to exit.")
                continue

            try:
                image = capture_image(camera_process)
            except RuntimeError as error:
                print(f"Capture failed: {error}")
                continue

            image_path = next_image_path(IMAGE_DIR)
            if cv2.imwrite(str(image_path), image):
                print(f"Captured image: {image_path}")
            else:
                print(f"Failed to save image: {image_path}")
    finally:
        camera_process.terminate()
        try:
            camera_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            camera_process.kill()
            camera_process.wait()
        CAPTURE_PATH.unlink(missing_ok=True)

    transfer_images(IMAGE_DIR, args.remote_host)


if __name__ == "__main__":
    main()