"""
ColorSnap: Extract and visualize dominant color palettes from images.

This package uses K-means clustering to extract the most prominent colors
from an image and provides various output formats including visualization,
JSON export, and console output.
"""

import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Union, Tuple, Dict, Any, Optional
import logging
import json
import os
import argparse
from pathlib import Path


class ColorSnap:
    """
    Extract and visualize color palettes from images using K-means clustering.
    """

    def __init__(
        self, output_dir: Optional[str] = None, log_level: int = logging.INFO
    ):
        """
        Initialize the ColorSnap extractor.

        Args:
            output_dir: Custom output directory. If None, one will be generated.
            log_level: Logging level (default: logging.INFO)
        """
        # Configure logging
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)
        self._output_dir = output_dir

    def extract_from_path(
        self, image_path: str, n_colors: int = 12
    ) -> Dict[str, Any]:
        """
        Extract color palette from an image file path.

        Args:
            image_path: Path to the image file
            n_colors: Number of colors to extract (default: 12)

        Returns:
            Dict containing extracted colors and normalized values
        """
        try:
            # Open and validate the image
            with Image.open(image_path) as img:
                img.verify()  # Verify that the file is a valid image

            # Reopen the file after verification
            img = Image.open(image_path)

            # Generate output directory if not specified
            if not self._output_dir:
                self._output_dir = self._generate_output_dir(image_path)

            # Process the image and return results
            return self.extract_from_image(img, n_colors)

        except Exception as e:
            self.logger.error(f"Error processing image '{image_path}': {e}")
            raise

    def extract_from_image(
        self, image: Image.Image, n_colors: int = 12
    ) -> Dict[str, Any]:
        """
        Extract color palette from a PIL Image object.

        Args:
            image: PIL Image object
            n_colors: Number of colors to extract (default: 12)

        Returns:
            Dict containing extracted colors and normalized values
        """
        try:
            # Process the image
            color_clusters = self._cluster_image_colors(
                image, n_colors=n_colors
            )
            self.logger.info(
                f"Successfully processed image with {n_colors} colors"
            )

            # Convert the color clusters to a list of RGB values
            colors = color_clusters.tolist()

            # Create normalized values (0-1 range for each RGB component)
            colors_norm = (color_clusters / 255).tolist()

            # Convert to hex values for easier use
            colors_hex = [mcolors.rgb2hex(tuple(c)) for c in colors_norm]

            return {
                "colors": colors,  # RGB values as integers (0-255)
                "colors_norm": colors_norm,  # Normalized RGB values (0-1)
                "colors_hex": colors_hex,  # Hex color codes
            }

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            raise

    def save_palette(
        self, palette: Dict[str, Any], filename: str = "palette.json"
    ) -> str:
        """
        Save color palette to a JSON file.

        Args:
            palette: Dictionary containing color palette data
            filename: Output filename (default: palette.json)

        Returns:
            Path to the saved file
        """
        try:
            # Ensure output directory exists
            os.makedirs(self._output_dir, exist_ok=True)

            # Full path to output file
            output_path = os.path.join(self._output_dir, filename)

            # Save the palette to a JSON file
            with open(output_path, "w") as f:
                json.dump(palette, f, indent=4)
                self.logger.info(f"Palette saved as '{output_path}'")

            return output_path

        except Exception as e:
            self.logger.error(f"Error saving palette: {e}")
            raise

    def print_palette(self, palette: Dict[str, Any]) -> None:
        """
        Print color palette to console.

        Args:
            palette: Dictionary containing color palette data
        """
        try:
            print("\nExtracted Colors:")
            for i, (rgb, hex_val) in enumerate(
                zip(palette["colors"], palette["colors_hex"]), 1
            ):
                print(
                    f"Color {str(i).rjust(2)}: RGB{tuple(rgb)} | {hex_val.upper()}"
                )
        except Exception as e:
            self.logger.error(f"Error printing palette: {e}")

    def create_visualization(
        self,
        image: Image.Image,
        palette: Dict[str, Any],
        figsize: Tuple[int, int] = (10, 6),
        show_values: bool = True,
        output_filename: str = "palette_visualization.png",
        title: Optional[str] = None,
    ) -> str:
        """
        Create a visualization of the image alongside its color palette.

        Args:
            image: Source image (PIL Image object)
            palette: Dictionary containing color palette data
            figsize: Figure size as (width, height) tuple
            show_values: Whether to display color values
            output_filename: Name of output file
            title: Title for the visualization (defaults to filename)

        Returns:
            Path to the saved visualization
        """
        try:
            # Input validation
            if not palette["colors"]:
                raise ValueError("Empty color palette provided")

            # Create figure with two subplots side by side
            fig = plt.figure(figsize=figsize)
            gs = plt.GridSpec(1, 2, width_ratios=[3, 1])

            # Left subplot - Image
            ax_img = fig.add_subplot(gs[0])
            ax_img.imshow(np.array(image))
            ax_img.axis("off")
            ax_img.set_title("Source Image", pad=20)

            # Title for the entire figure
            if title:
                fig.suptitle(title, fontsize=16, y=0.98)

            # Right subplot - Color Palette
            ax_palette = fig.add_subplot(gs[1])
            ax_palette.set_title("Extracted Colors", pad=20)

            # Calculate rectangle positions
            n_colors = len(palette["colors"])
            rect_height = 1 / n_colors

            # Plot color rectangles
            for i, (color, hex_color) in enumerate(
                zip(palette["colors_norm"], palette["colors_hex"])
            ):
                # Calculate rectangle position
                bottom = 1 - (i + 1) * rect_height

                # Add color rectangle
                rect = plt.Rectangle(
                    (0, bottom),
                    1,
                    rect_height,
                    facecolor=color,
                    edgecolor="white",
                )
                ax_palette.add_patch(rect)

                if show_values:
                    # Calculate text color based on background brightness
                    brightness = np.mean(color)
                    text_color = "white" if brightness < 0.5 else "black"

                    # Add color value text
                    ax_palette.text(
                        0.5,
                        bottom + rect_height / 2,
                        hex_color.upper(),
                        horizontalalignment="center",
                        verticalalignment="center",
                        color=text_color,
                        fontsize=12,
                    )

            # Set palette subplot properties
            ax_palette.set_xlim(0, 1)
            ax_palette.set_ylim(0, 1)
            ax_palette.axis("off")

            # Adjust layout
            plt.tight_layout()

            # Ensure output directory exists
            os.makedirs(self._output_dir, exist_ok=True)

            # Save the visualization
            output_path = os.path.join(self._output_dir, output_filename)
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Visualization saved as '{output_path}'")

            plt.close(fig)
            return output_path

        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            raise

    def process_and_save(
        self,
        image_path: str,
        n_colors: int = 12,
        create_viz: bool = True,
        save_json: bool = True,
        print_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Process an image, extract colors, and save outputs.

        This is a convenience method that combines multiple steps.

        Args:
            image_path: Path to the image file
            n_colors: Number of colors to extract
            create_viz: Whether to create and save visualization
            save_json: Whether to save palette as JSON
            print_results: Whether to print results to console

        Returns:
            Dict with results including palette and output paths
        """
        # Extract palette from image
        try:
            # Process the image
            palette = self.extract_from_path(image_path, n_colors)

            results = {"palette": palette, "outputs": {}}

            # Load image for visualization
            image = Image.open(image_path)

            # Print to console if requested
            if print_results:
                self.print_palette(palette)

            # Save JSON if requested
            if save_json:
                json_path = self.save_palette(palette)
                results["outputs"]["json"] = json_path

            # Create visualization if requested
            if create_viz:
                # Generate output filename based on input
                viz_filename = f"{Path(image_path).stem}_palette.png"
                viz_path = self.create_visualization(
                    image, palette, output_filename=viz_filename
                )
                results["outputs"]["visualization"] = viz_path

            return results

        except Exception as e:
            self.logger.error(f"Error in process_and_save: {e}")
            raise

    def _cluster_image_colors(
        self, image: Image.Image, n_colors: int = 12
    ) -> np.ndarray:
        """
        Cluster image RGB values using KMeans.

        Args:
            image: PIL Image object
            n_colors: Number of color clusters to extract

        Returns:
            NumPy array of RGB color values (shape: n_colors, 3)
        """
        # Convert to RGB mode if not already
        image = image.convert("RGB")

        # Resize image to improve performance
        # Using a fixed size for consistent results
        # The resize is optional but helps with large images
        width, height = image.size
        max_dimension = 400  # Max dimension for processing

        # Only resize if the image is larger than max_dimension
        if width > max_dimension or height > max_dimension:
            # Calculate scale factor to maintain aspect ratio
            scale = max_dimension / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.LANCZOS)
            self.logger.info(
                f"Resized image from {width}x{height} to {new_size[0]}x{new_size[1]} for processing"
            )

        # Convert image to numpy array and reshape
        image_array = np.array(image).reshape(
            -1, 3
        )  # Flatten to (num_pixels, 3)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(image_array)

        # Get cluster centers (these are the dominant colors)
        clusters = kmeans.cluster_centers_.astype(int)

        # Sort colors by frequency (pixel count in each cluster)
        labels = kmeans.labels_
        label_counts = np.bincount(labels)

        # Get indices of clusters ordered by frequency (most frequent first)
        # This makes the palette more representative of the image
        ordered_indices = np.argsort(-label_counts)
        ordered_clusters = clusters[ordered_indices]

        return ordered_clusters

    def _generate_output_dir(self, image_path: str) -> str:
        """
        Generate output directory name based on image path.

        Args:
            image_path: Path to the image file

        Returns:
            Path to the output directory
        """
        # Get the image filename without extension
        image_name = Path(image_path).stem

        # Create output directory name
        output_dir = f"colorsnap-{image_name}"

        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"ColorSnap exports will be saved in '{output_dir}'")

        return output_dir

    @property
    def output_dir(self) -> str:
        """Get the current output directory."""
        return self._output_dir


def main():
    """Command-line interface for ColorSnap."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="ColorSnap: Extract color palette from an image"
    )
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "-n",
        "--n-colors",
        type=int,
        default=12,
        help="Number of colors to extract (default: 12)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: auto-generated)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip creating visualization",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving JSON palette",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Skip printing results to console",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Determine log level
    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Initialize ColorSnap
    extractor = ColorSnap(output_dir=args.output_dir, log_level=log_level)

    # Try to find the image
    image_path = args.image
    if not os.path.exists(image_path):
        # Check in the data directory
        data_path = os.path.join("samples", args.image)
        if os.path.exists(data_path):
            image_path = data_path
            print(f"Using image from data directory: {image_path}")
        else:
            print(
                f"Error: Image not found at '{args.image}' or in data directory"
            )
            return 1

    try:
        # Process the image and save results
        results = extractor.process_and_save(
            image_path,
            n_colors=args.n_colors,
            create_viz=not args.no_viz,
            save_json=not args.no_json,
            print_results=not args.no_print,
        )

        # Print output information
        print(f"\nColorSnap successfully processed '{image_path}'")
        print(f"Results saved to '{extractor.output_dir}'")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
