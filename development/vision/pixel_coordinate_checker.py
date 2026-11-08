"""Display detected board corners with an interactive pixel-coordinate tooltip."""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import config

from vision.vis_board import orient_image, process_image, undistort_image


IMAGE_PATHS = [
    config.VISION_ASSETS_DIR / "test grids" / "weblarge.jpg",
    # config.VISION_ASSETS_DIR / "test grids" / "websmall.jpg",
    # config.VISION_ASSETS_DIR / "test grids" / "grid6.jpg",
    # config.VISION_ASSETS_DIR / "test grids" / "grid7.jpg",
    # config.VISION_ASSETS_DIR / "test grids" / "grid8.jpg",
    # config.VISION_ASSETS_DIR / "test grids" / "grid9.jpg",
    # config.VISION_ASSETS_DIR / "test grids" / "thicc.jpg",
    # config.VISION_ASSETS_DIR / "test grids" / "krazydad.jpg",
    # config.VISION_ASSETS_DIR / "test grids" / "camera_test_good.jpg",
    # config.VISION_ASSETS_DIR / "test grids" / "camera_test_bad.jpg",
    # config.VISION_ASSETS_DIR / "test grids" / "demo.png",
]


def show_coordinate_checker(image_path: Path) -> None:
    """Detect and display corners for one image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    board = process_image(image)
    display_image = orient_image(undistort_image(image))
    display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

    figure, axes = plt.subplots()
    axes.imshow(display_image, origin="upper")
    axes.plot(
        board.corners[:, 0],
        board.corners[:, 1],
        "o-",
        color="red",
        linewidth=2,
        markersize=7,
    )

    for index, (x, y) in enumerate(board.corners):
        axes.annotate(
            f"{index}: ({x:.1f}, {y:.1f})",
            (x, y),
            xytext=(8, 8),
            textcoords="offset points",
            color="yellow",
        )

    tooltip = axes.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "black", "alpha": 0.8},
        color="white",
    )
    tooltip.set_visible(False)

    def update_tooltip(event) -> None:
        if event.inaxes is not axes or event.xdata is None or event.ydata is None:
            tooltip.set_visible(False)
            figure.canvas.draw_idle()
            return

        tooltip.xy = (event.xdata, event.ydata)
        tooltip.set_text(f"x={event.xdata:.1f}px, y={event.ydata:.1f}px")
        tooltip.set_visible(True)
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("motion_notify_event", update_tooltip)
    axes.set_title(f"Detected corners: {image_path.name}")
    axes.set_xlabel("x (pixels)")
    axes.set_ylabel("y (pixels)")
    figure.tight_layout()
    plt.show()


def main() -> None:
    """Show the configured images one at a time."""
    for image_path in IMAGE_PATHS:
        show_coordinate_checker(image_path)


if __name__ == "__main__":
    main()
