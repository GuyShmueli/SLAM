"""
slam/frontend/io/image_sequence.py

Objective:
    Provide an abstraction for loading synchronized stereo image pairs from disk,
    handling file naming conventions and dataset directory structure so downstream
    modules can simply index frames without worrying about paths or file I/O.
"""
from pathlib import Path
import cv2

class ImageSequence:
    """
    Manages a sequence of stereo image pairs stored in two separate directories.

    ---
    Attributes:
    left_paths     list of all left-images' paths
    right_paths    list of all right-images' paths
    """
    def __init__(
        self,
        base_path: Path,
        left_dir: str = "image_0",
        right_dir: str = "image_1",
        extension: str = ".png"
    ):
        """
        Args:
            base_path: Path to the sequence folder containing two subdirectories.
            left_dir:  Name of the subdirectory for left images.
            right_dir: Name of the subdirectory for right images.
            extension: File extension for the images ('.png', '.jpg' etc.).
        """
        base = Path(base_path)
        if not base.is_dir():
            raise FileNotFoundError(f"Base directory not found: {base}")

        self.left_paths = sorted((base / left_dir).glob(f"*{extension}"))
        self.right_paths = sorted((base / right_dir).glob(f"*{extension}"))

        if len(self.left_paths) != len(self.right_paths):
            raise ValueError(
                f"Mismatched number of left ({len(self.left_paths)}) vs right ({len(self.right_paths)}) images"
            )


    def __len__(self):
        """ Return the number of images in this sequence. """
        return len(self.left_paths)


    def __getitem__(self, idx):
        """
        Load and return the stereo pair at the given index.
        Returns:
            (left_image, right_image) both as numpy arrays (grayscale).
        """
        if idx < 0 or idx >= len(self):      # utilizes __len__
            raise IndexError(f"Index {idx} out of range")

        left_path = self.left_paths[idx]
        right_path = self.right_paths[idx]
        left_img = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)

        if left_img is None or right_img is None:
            raise IOError(f"Could not read images at index {idx}: "
                          f"{left_path, right_path}")
        return left_img, right_img


    def __iter__(self):
        """
        Allow iteration over all frames: for left, right in sequence.
        'yield' is used to create a generator, which produces one frame
        at a time, on demand, instead of building a giant list of all frames.
        """
        for idx in range(len(self)):
            yield self[idx]           # utilizes __getitem__


