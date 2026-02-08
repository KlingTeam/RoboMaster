from PIL import Image
import os

def resize_images_in_folder(folder_path, output_folder, size=(640, 480)):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Build the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                # Open the image
                with Image.open(file_path) as img:
                    # Resize the image
                    resized_img = img.resize(size)
                    
                    # Save the resized image to the output folder
                    resized_img.save(os.path.join(output_folder, filename))
                    print(f"Resized and saved {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
resize_images_in_folder('inthewild_img', 'output')
