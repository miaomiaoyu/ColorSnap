"""
Examples of how to use the ColorSnap package for color extraction.
"""

from colorsnap import ColorSnap
from PIL import Image


def basic_example():
    """Basic usage: process image and save all outputs."""

    # Initialize ColorSnap
    extractor = ColorSnap(output_dir="data/outputs")

    # Process an image with default settings (12 colors)
    results = extractor.process_and_save("data/samples/img-03.jpg")

    # Print the hex color codes
    print("Extracted hex colors:")
    for hex_color in results["palette"]["colors_hex"]:
        print(f"  {hex_color}")

    print(f"\nOutputs saved to: {extractor.output_dir}")


def custom_colors_example():
    """Extract a custom number of colors."""

    # Initialize ColorSnap with a specific output directory
    extractor = ColorSnap(output_dir="data/outputs")

    # Extract 8 colors instead of the default 12
    palette = extractor.extract_from_path("data/samples/img-04.jpg", n_colors=8)

    # Print RGB values
    print("RGB Values:")
    for rgb in palette["colors"]:
        print(f"  RGB{tuple(rgb)}")

    # Save only the JSON
    extractor.save_palette(palette, filename="custom_palette.json")


def programmatic_example():
    """Use ColorSnap in a programmatic workflow."""

    # Open an image with PIL
    image = Image.open("data/samples/img-05.jpg")

    # Initialize ColorSnap
    extractor = ColorSnap(output_dir="data/outputs")

    # Extract colors directly from the PIL Image
    palette = extractor.extract_from_image(image, n_colors=6)

    # Create a visualization with a custom title
    extractor.create_visualization(
        image,
        palette,
        figsize=(12, 8),
        output_filename="product_palette.png",
        title="Product Color Analysis",
    )

    # Work with the extracted colors in your application
    for i, (rgb, hex_color) in enumerate(
        zip(palette["colors"], palette["colors_hex"])
    ):
        print(f"Color {i + 1}: {hex_color} - RGB{tuple(rgb)}")

        # Example: Use these colors in your application
        # my_app.apply_color(hex_color)


def batch_processing_example():
    """Process multiple images in batch."""

    import os

    # List of images to process
    image_files = [
        "data/samples/img-01.jpg",
        "data/samples/img-02.jpg",
        "data/samples/img-03.jpg",
    ]

    # Initialize ColorSnap
    extractor = ColorSnap(output_dir="data/outputs")

    # Process each image
    for img_path in image_files:
        print(f"Processing: {img_path}")

        # Get the filename without extension for the output
        filename = os.path.splitext(os.path.basename(img_path))[0]

        # Extract colors
        palette = extractor.extract_from_path(img_path, n_colors=10)

        # Save JSON with custom filename
        extractor.save_palette(palette, filename=f"{filename}.json")

        # Print a summary
        print(f"  Extracted {len(palette['colors'])} colors")
        print(f"  Primary color: {palette['colors_hex'][0]}")
        print()


if __name__ == "__main__":
    print("Running ColorSnap examples:")
    print("\n1. Basic Example")
    print("----------------")
    basic_example()

    print("\n2. Custom Colors Example")
    print("-----------------------")
    custom_colors_example()

    print("\n3. Programmatic Example")
    print("-----------------------")
    programmatic_example()

    print("\n4. Batch Processing Example")
    print("---------------------------")
    batch_processing_example()
