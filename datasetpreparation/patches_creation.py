from PIL import Image
import os

# Define paths
input_dir = 'data/TCGA_GBMLGG/all_st'  # Directory containing images
output_dir = 'data/TCGA_GBMLGG/patches'
os.makedirs(output_dir, exist_ok=True)

# List all image files in the directory
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process each image
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    img = Image.open(image_path)

    # Extract image name without extension
    image_name = os.path.splitext(image_file)[0]

    print("Patch Creation for ", image_name)
    # Define patch size
    patch_width = img.width // 2
    patch_height = img.height // 2

    # Generate 4 patches
    coordinates = [
        (0, 0),
        (patch_width, 0),
        (0, patch_height),
        (patch_width, patch_height)
    ]

    for idx, (x, y) in enumerate(coordinates):
        patch = img.crop((x, y, x + patch_width, y + patch_height))
        patch_filename = os.path.join(output_dir, f'{image_name}_{x}_{y}.png')
        patch.save(patch_filename)

print("Generated patches for all images.")
