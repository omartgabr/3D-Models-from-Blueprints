# computer vision
import fitz
import cv2

# miscellaneous
import numpy as np
import os
import shutil
import json


#################
# UPDATE JSON AND SAVE IMAGE
#################
def create_json_files(metadata, output_folder_path):
    """
    Creates JSON files for each image based on the provided metadata and saves them to the corresponding output folder.
    
    Parameters:
    - metadata: A dictionary containing metadata for each image, including 'scale', 'floor', 'elevation', and 'corners'.
    - output_folder_path: The root directory where the JSON files will be saved.
    """
    
    # Loop through the metadata and create JSON files for each image
    for filename, img_metadata in metadata.items():
        label = img_metadata.get('label')
        page_num = img_metadata.get('page_num')

        # Corresponding JSON filename (e.g., architectural_4_info.json)
        json_filename = f"{label}_{page_num}_info.json"
        json_file_path = os.path.join(output_folder_path, label, json_filename)

        # Create the folder if it doesn't exist
        os.makedirs(os.path.join(output_folder_path, label), exist_ok=True)

        # Formatted corners (if corners are part of metadata)
        formatted_corners = [f"({corner[0]}, {corner[1]})" for corner in img_metadata.get('corners', [])]

        # Define the new image_info structure using metadata
        image_info = {
            filename: {
                "file_type": label,
                "scale": img_metadata.get('scale', 'N/A'),
                "floor": img_metadata.get('floor', 'N/A'),
                "elevation": img_metadata.get('elevation', 'N/A'),
                "boundary_corners": formatted_corners,
                "original_page_in_pdf": page_num
            }
        }

        # Create the JSON file
        with open(json_file_path, 'w') as f:
            json.dump(image_info, f, indent=4)

        print(f"Metadata created for {filename} in {json_file_path}.")

#################
# EXTRACT CONTOUR BY INDEX
#################
def extract_contour_by_index(img, index=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Ensure that the index is within the range of available contours
    if index < len(sorted_contours):
        return sorted_contours[index]
    else:
        print(f"Contour index {index} out of range. Only {len(sorted_contours)} contours found.")
        return None


#################
# DELETE FILES IN FOLDER
#################
# deletes all the files in the specified folder to clear
# before adding newly labeled images
def delete_files_in_folder(folder):
    """
    Deletes all files in the given folder and its subfolders without deleting the folders.
    """
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    print(f"{file_path} has been deleted")
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


#################
# PERFORM MATCHING
#################
# compares the input image against the various templates
def perform_matching(image, templates):
    best_label = None
    best_score = 0

    # Loop through the templates and determine scales for best performance
    for label, template, (w, h) in templates:
        local_best_score = 0  # Initialize local_best_score for each template

        scales = [0.5, 0.75, 1, 1.25, 1.5]
        for scale in scales:
            # Resize the template based on the scale
            resized_template = cv2.resize(template, None, fx=scale, fy=scale)
            
            # Check if resized template is larger than the image
            if resized_template.shape[0] > image.shape[0] or resized_template.shape[1] > image.shape[1]:
                # Resize template to fit within the image while maintaining aspect ratio
                aspect_ratio = resized_template.shape[1] / resized_template.shape[0]
                if resized_template.shape[1] > resized_template.shape[0]:  # wider
                    new_width = image.shape[1]
                    new_height = int(new_width / aspect_ratio)
                else:  # taller
                    new_height = image.shape[0]
                    new_width = int(new_height * aspect_ratio)

                resized_template = cv2.resize(resized_template, (new_width, new_height))

            # Perform template matching
            res = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > local_best_score:
                local_best_score = max_val

        if local_best_score > best_score:
            best_score = local_best_score
            best_label = label

    return best_label, best_score


#################
# EXTRACT FLOORPLAN LABEL
#################
def extract_floorplan_label(base_folder_path, templates):
    """
    Loads images and templates, processes and matches each image to the most fitting template,
    and returns the page number and label (category) for each image.
    """
    # Compile list of templates that will be used for matching
    prepared_templates = []
    metadata = {}
    
    for label, template_path in templates.items():
        template_image = cv2.imread(template_path, 0)  # Read as grayscale

        if template_image is None:
            print(f"Failed to load template image for {label} from {template_path}")
            continue

        # Convert to grayscale if needed and preprocess
        template_image = template_image.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        template_image = clahe.apply(template_image)

        prepared_templates.append((label, template_image, template_image.shape[::-1]))

    # Loop through each image in the folder to process, extract page_num and label
    for filename in os.listdir(base_folder_path):
        file_path = os.path.join(base_folder_path, filename)
        img = cv2.imread(file_path)

        if img is None:
            print(f"Failed to load image: {file_path}")
            continue

        # Extract page number from the filename
        page_num = int(filename.split('_')[1].split('.')[0])

        # Perform template matching using perform_matching function on the original image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        best_label, best_score = perform_matching(gray_img, prepared_templates)

        # Adjusting the threshold parameter for matching
        threshold = 0.75
        if best_score >= threshold:
            print(f"Template matched as {best_label} with score {best_score}")
            print(f"Categorizing as {best_label} for {filename}")
            metadata[filename] = {'page_num': page_num, 'label': best_label}
        else:
            print(f"Categorizing as Uncategorized for {filename}")
            metadata[filename] = {'page_num': page_num, 'label': 'Uncategorized'}

    return metadata


def outer_edge_detection(base_folder_path, output_folder_path, metadata):
    """
    Loops through images in the base_folder_path, applies edge detection,
    saves the processed images to output_folder_path, and returns updated metadata.
    """
    # Traverse through all subdirectories in the base folder path
    for subdir, dirs, files in os.walk(base_folder_path):
        for file in files:
            # Process only image files (assuming they are .png)
            if file.endswith(".png"):
                image_path = os.path.join(subdir, file)
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Failed to load image: {file}, skipping.")
                    continue

                # Get corresponding metadata for the current image (already populated from extract_floorplan_label)
                if file in metadata:
                    img_metadata = metadata[file]

                    # Perform outer edge detection for this image
                    largest_contour = extract_contour_by_index(img, index=0)
                    if largest_contour is None:
                        print(f"Failed to find largest contour in {file}, skipping.")
                        continue

                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cropped_image = img[y:y + h - 10, x:x + w - 10]
                    second_largest_contour = extract_contour_by_index(cropped_image, index=0)
                    if second_largest_contour is None:
                        print(f"Failed to find second largest contour in {file}, skipping.")
                        continue

                    x2, y2, w2, h2 = cv2.boundingRect(second_largest_contour)
                    cv2.rectangle(cropped_image, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                    corners = [(x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2)]
                    layout_image = cropped_image[y2:y2 + h2 - 10, x2:x2 + w2 - 10]

                    # Update metadata with corner information without overwriting existing data
                    img_metadata['corners'] = corners

                    # Save the processed image in the appropriate output folder
                    best_label = img_metadata['label']
                    output_path = os.path.join(output_folder_path, best_label, file)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, layout_image)
                    print(f"Categorizing as {best_label}, saving layout image to: {output_path}")

                else:
                    print(f"No metadata found for {file}, skipping.")

    # Return updated metadata
    return metadata






