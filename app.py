from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import logging
import io

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cluster_image_rgb_values(image, n_colors=10):
    """Cluster image RGB values using KMeans."""
    image = image.convert("RGB")
    # Resize image to improve performance (optional)
    image = image.resize((200, 200))  # Resize to 200x200 pixels
    image_array = np.array(image).reshape(-1, 3)  # Flatten to (num_pixels, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=1, n_init=10)
    kmeans.fit(image_array)
    clusters = kmeans.cluster_centers_.astype(int)
    return clusters

@app.route("/")
def home():
    """Serve the index.html file."""
    return render_template("index.html")

@app.route("/process_image", methods=["POST"])
def process_image():
    """Process the uploaded image and return color clusters."""
    if "image" not in request.files:
        logger.error("No image uploaded")
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    n_colors = request.form.get("n_colors", default=10, type=int)

    # Validate n_colors
    if n_colors < 2 or n_colors > 20:
        logger.error(f"Invalid n_colors value: {n_colors}")
        return jsonify({"error": "n_colors must be between 2 and 20"}), 400

    try:
        # Open and validate the image
        image = Image.open(file)
        image.verify()  # Verify that the file is a valid image
        image = Image.open(file)  # Reopen the file after verification

        # Process the image
        color_clusters = cluster_image_rgb_values(image, n_colors=n_colors)
        logger.info(f"Successfully processed image with {n_colors} colors")

        # Return JSON response
        return jsonify({"colors": color_clusters.tolist()})
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/download_palette", methods=["POST"])
def download_palette():
    """Generate and return a downloadable JSON file of the color palette."""
    if "image" not in request.files:
        logger.error("No image uploaded")
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    n_colors = request.form.get("n_colors", default=10, type=int)

    try:
        # Open and validate the image
        image = Image.open(file)
        image.verify()
        image = Image.open(file)

        # Process the image
        color_clusters = cluster_image_rgb_values(image, n_colors=n_colors)
        palette = {"colors": color_clusters.tolist()}

        # Create a JSON file in memory
        json_file = io.BytesIO()
        json_file.write(jsonify(palette).data)
        json_file.seek(0)

        # Return the file as a downloadable response
        return send_file(
            json_file,
            mimetype="application/json",
            as_attachment=True,
            download_name="palette.json"
        )
    except Exception as e:
        logger.error(f"Error generating palette: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)