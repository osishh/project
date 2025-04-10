from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import shutil
import subprocess
import base64
from PIL import Image
from io import BytesIO
import uuid

# Setup Flask with static files from Flutter web build
app = Flask(__name__, static_folder='build/web', static_url_path='')
CORS(app)  # Enable CORS for cross-origin requests

# Directories
dataset_dir = "test_dataset"
input_subdir = os.path.join(dataset_dir, "testA")  # This is what the model expects
output_dir = "./ablation/enlightening/test_200/images"

# Ensure input/output folders exist
os.makedirs(input_subdir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_b64 = data.get('image')
    if not image_b64:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400

    # ✅ Clean the input directory
    shutil.rmtree(input_subdir)
    os.makedirs(input_subdir, exist_ok=True)

    # ✅ Clean the output directory
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            file_path = os.path.join(output_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Decode and save the uploaded image
    image_data = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    unique_filename = f"{uuid.uuid4().hex}.jpg"
    input_path = os.path.join(input_subdir, unique_filename)
    image.save(input_path)

    # Run the image enhancement model
    subprocess.run([
        "python", "predict.py",
        "--dataroot", dataset_dir,
        "--name", "enlightening",
        "--model", "single",
        "--which_direction", "AtoB",
        "--no_dropout",
        "--dataset_mode", "unaligned",
        "--which_model_netG", "sid_unet_resize",
        "--skip", "1",
        "--use_norm", "1",
        "--use_wgan", "0",
        "--self_attention",
        "--times_residual",
        "--instance_norm", "0",
        "--resize_or_crop", "no",
        "--which_epoch", "200"
    ])

    # Find and return the enhanced image
    output_files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not output_files:
        return jsonify({'status': 'error', 'message': 'No output image generated'}), 500

    latest_output = max([os.path.join(output_dir, f) for f in output_files], key=os.path.getctime)
    with open(latest_output, "rb") as img_file:
        enhanced_b64 = base64.b64encode(img_file.read()).decode('utf-8')

    return jsonify({'status': 'success', 'enhanced_image': enhanced_b64})

# ✅ Handle Flutter Web Routes (SPA fallback)
@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

# ✅ Run the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
