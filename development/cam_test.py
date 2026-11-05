
"""Capture image on the pi, which will be ssh'd into, then send it back and display it on the computer"""

from pathlib import Path
import subprocess

import cv2
import numpy as np


OUTPUT_PATH = Path(__file__).with_name("camera_test.jpg")
REMOTE_USER = "haydnjohns"
REMOTE_HOST = "192.168.1.118"
REMOTE_PATH = "/Users/haydnjohns/Downloads/camera_test.jpg"


def run_remote_command(command: list[str], description: str) -> None:
    """Run a remote command and print enough output to diagnose failures."""
    print(f"[debug] {description}: {' '.join(command)}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True)

    if result.stdout:
        print(f"[debug] stdout:\n{result.stdout.rstrip()}", flush=True)
    if result.stderr:
        print(f"[debug] stderr:\n{result.stderr.rstrip()}", flush=True)
    print(f"[debug] exit code: {result.returncode}", flush=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed with exit code {result.returncode}"
        )


# --------------------------------------------------------------------------
# Image capture
# --------------------------------------------------------------------------

def capture_image(width: int = 2592, height: int = 1944) -> np.ndarray:
    """Capture an image using the Raspberry Pi camera."""
    result = subprocess.run(
        [
            "rpicam-jpeg",
            "-o", "-",
            "--width", str(width),
            "--height", str(height),
            "--nopreview",
        ],
        capture_output=True,
        check=True
    )

    image = cv2.imdecode(
        np.frombuffer(result.stdout, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    if image is None:
        raise RuntimeError("Failed to decode captured image")
    return image

def main() -> None:
    """Capture an image, save it locally, and display it."""
    image = capture_image()

    if not cv2.imwrite(str(OUTPUT_PATH), image):
        raise RuntimeError(f"Failed to save captured image to {OUTPUT_PATH}")

    print(f"[debug] Captured image saved to {OUTPUT_PATH}", flush=True)

    remote = f"{REMOTE_USER}@{REMOTE_HOST}"
    run_remote_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            remote,
            "printf 'Connected as '; whoami; printf ' on '; hostname",
        ],
        "SSH connectivity check",
    )
    run_remote_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            remote,
            "test -d /Users/haydnjohns/Downloads",
        ],
        "Remote destination check",
    )
    run_remote_command(
        [
            "scp",
            str(OUTPUT_PATH),
            f"{remote}:{REMOTE_PATH}",
        ],
        "Image transfer",
    )
    run_remote_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            remote,
            "test -f " + REMOTE_PATH,
        ],
        "Remote file check",
    )

    run_remote_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            remote,
            "open",
            REMOTE_PATH,
        ],
        "Remote image open",
    )
    print("[debug] Image opened on the computer", flush=True)


if __name__ == "__main__":
    main()
