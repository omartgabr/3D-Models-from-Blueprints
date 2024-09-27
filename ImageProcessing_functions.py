# computer vision
import fitz
import cv2

# miscellaneous
import numpy as np
import os



#################
# 2
#################
def create_folder_structure(base_path):
    """
    create the main folder and subfolders for labeling floorplans
    """
    main_folder = os.path.join(base_path, 'Labeled_Floorplans')
    subfolders = ['Architectural', 'Structural', 'Mechanical', 'Plumbing', 'Uncategorized']
    
    # create main folder if it doesn't exist
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    # create subfolders
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
    return main_folder

#################
# 3
#################
def delete_files_in_folder(folder):
    """
    deletes all the files in the specified folder
    """
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
# 4
#################
def process_image(image):
    '''
    process the template images for proper matching and labeling
    '''
    if image is None:
        print(f"Failed to load image")
        return None

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    # Gaussian blur for noise reduction
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


#################
# 5
#################
def perform_matching(image, templates):
    '''
    template matching logic
    '''
    best_label = None
    best_score = 0

    for label, template, (w, h) in templates:
        scales = [0.5, 0.75, 1, 1.25, 1.5]
        local_best_score = 0

        for scale in scales:
            resized_template = cv2.resize(template, None, fx=scale, fy=scale)
            res = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > local_best_score:
                local_best_score = max_val

        if local_best_score > best_score:
            best_score = local_best_score
            best_label = label

    return best_label, best_score


#################
# 6
#################
def extract_floorplan_label(raw_images_folder_path, templates, output_folder_path):
    '''
    Creates a folder with subfolders to classify the floorplans, then
    performs template matching using the preprocessed template features to
    sort the images accordingly.
    '''
    # Create folders for labeling inside the base path
    main_folder = create_folder_structure(output_folder_path)

    # Path to uncategorized folder (ensure it's created in the correct base_path)
    uncategorized_folder = os.path.join(main_folder, 'Uncategorized')

    # Ensure labeled folders and uncategorized folders are clean
    for label in templates.keys():
        label_folder = os.path.join(main_folder, label)
        if os.path.exists(label_folder):
            delete_files_in_folder(label_folder)
        else:
            os.makedirs(label_folder)

    if os.path.exists(uncategorized_folder):
        delete_files_in_folder(uncategorized_folder)
    else:
        os.makedirs(uncategorized_folder)

    # Prepare templates for matching
    prepared_templates = []
    for label, template_path in templates.items():
        template_image = cv2.imread(template_path, 0)  # Read as grayscale
        if template_image is None:
            print(f"Failed to load template for {label} from {template_path}")
        else:
            print(f"Loaded template for {label}, shape: {template_image.shape}")

        processed_template = process_image(template_image)
        if processed_template is not None:
            prepared_templates.append((label, processed_template, processed_template.shape[::-1]))

    for filename in os.listdir(raw_images_folder_path):
        file_path = os.path.join(raw_images_folder_path, filename)
        img = cv2.imread(file_path)
        if img is None:
            print(f"Failed to load image: {file_path}")
        else:
            print(f"Loaded image {filename}, shape: {img.shape}")

        categorized = False

        # Convert to grayscale and process the image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = process_image(gray_img)

        # Perform template matching
        best_label, best_score = perform_matching(gray_img, prepared_templates)

        # Debug: Print the best score for the current image
        print(f"Best score for image {filename}: {best_score}")

        # Set threshold for matching
        threshold = 0.65  # Adjust threshold as needed
        if best_score >= threshold:
            output_path = os.path.join(main_folder, best_label, filename)
            print(f"Categorizing as {best_label} with score {best_score}, saving to: {output_path}")
            cv2.imwrite(output_path, img)
            categorized = True

        # If the image was not categorized, save it to the 'Uncategorized' folder
        if not categorized:
            uncategorized_path = os.path.join(uncategorized_folder, filename)
            print(f"Categorizing as Uncategorized: {uncategorized_path}")
            cv2.imwrite(uncategorized_path, img)


#################
# 7
#################












