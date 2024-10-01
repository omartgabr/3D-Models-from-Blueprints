# computer vision
import fitz
import cv2

# miscellaneous
import numpy as np
import os
import shutil


#################
# 1
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
# 2
#################
# deletes all the files in the specified folder to clear
# before adding newly labeled images
def delete_files_in_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"{filename} has been deleted")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


#################
# 3
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
# 4
#################
def extract_floorplan_label(base_folder_path, templates, output_folder_path):
    """
    Loads images and templates, processes and matches each image to the most fitting template,
    then applies contour extraction twice (first for the largest contour, then for the layout),
    and saves the layout to the appropriate folder based on the matching template label.
    """

    # Ensure the output folder path exists, or create it if necessary
    if not os.path.exists(output_folder_path):
        print(f"Output folder path '{output_folder_path}' does not exist. Creating it now.")
        os.makedirs(output_folder_path)

    # Create or clean labeled folders for each label in the templates
    for label in templates.keys():
        label_folder = os.path.join(output_folder_path, label)
        if os.path.exists(label_folder):
            delete_files_in_folder(label_folder)
        else:
            os.makedirs(label_folder)

    # Automatically find or create the 'Uncategorized' folder
    uncategorized_folder = os.path.join(output_folder_path, 'Uncategorized')
    if os.path.exists(uncategorized_folder):
        delete_files_in_folder(uncategorized_folder)
    else:
        os.makedirs(uncategorized_folder)

    # To store scores for each label
    score_tracker = {label: [] for label in templates.keys()}

    # Compile list of templates that will be used for matching
    prepared_templates = []
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

    # Loop through each image in the folder to process, extract contours, and label
    for filename in os.listdir(base_folder_path):
        file_path = os.path.join(base_folder_path, filename)
        img = cv2.imread(file_path)

        if img is None:
            print(f"Failed to load image: {file_path}")
            continue

        # Initialize categorized to False for each image
        categorized = False

        # Perform template matching using perform_matching function on the original image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        best_label, best_score = perform_matching(gray_img, prepared_templates)

        # Adjusting the threshold parameter for matching
        threshold = 0.75
        if best_score >= threshold:
            print(f"Template matched as {best_label} with score {best_score}")

            # Step 1: Extract the largest contour (likely outer boundary)
            largest_contour = extract_contour_by_index(img, index=0)
            if largest_contour is None:
                print(f"Failed to find largest contour in {filename}, skipping.")
                continue

            # Step 2: Get bounding rectangle for the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_image = img[y:y + h - 10, x:x + w - 10]  # Apply the same margin as in Colab

            # Step 3: Extract the second largest contour within the cropped image (likely floorplan layout)
            second_largest_contour = extract_contour_by_index(cropped_image, index=0)
            if second_largest_contour is None:
                print(f"Failed to find second largest contour in {filename}, skipping.")
                continue

            # Step 4: Get bounding rectangle for the second largest contour (floorplan layout)
            x2, y2, w2, h2 = cv2.boundingRect(second_largest_contour)
            layout_image = cropped_image[y2:y2 + h2 - 10, x2:x2 + w2 - 10]  # Apply margin for the layout crop

            # Save the layout image to the corresponding labeled folder
            output_path = os.path.join(output_folder_path, best_label, filename)
            print(f"Categorizing as {best_label}, saving layout image to: {output_path}")
            cv2.imwrite(output_path, layout_image)  # Save layout image to output folder
            categorized = True  # Mark the image as categorized

        # If the image was not categorized, save it to the 'Uncategorized' folder
        if not categorized:
            uncategorized_path = os.path.join(uncategorized_folder, filename)
            print(f"Categorizing as Uncategorized: {uncategorized_path}")
            cv2.imwrite(uncategorized_path, img)

    # Calculate and return the average score for each label
    average_scores = {label: (sum(scores) / len(scores)) if scores else 0 for label, scores in score_tracker.items()}
    return average_scores











