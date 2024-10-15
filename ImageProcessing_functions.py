# computer vision
import fitz
import cv2
import pytesseract

# miscellaneous
import numpy as np
import os
import shutil
import json
import re

import ImageProcessing_functions as IPF

##################################
# CREATE PROCESSED DATA FOLDER
##################################
# deletes all the files in the specified folder to clear
# before adding newly labeled images
def create_processed_data_folder(folder_name):
    """
    Creates a folder with the given name. If a folder with the same name already exists, 
    it will be overwritten by deleting the existing folder and creating a new one.
    
    Returns the relative path of the created folder.
    """
    if os.path.exists(folder_name):
        # Remove the existing folder and all its contents
        shutil.rmtree(folder_name)
        print(f"Existing folder '{folder_name}' has been deleted.")

    # Create a new folder
    os.makedirs(folder_name)
    print(f"New folder '{folder_name}' has been created.")
    
    # Return the relative path of the folder
    return os.path.relpath(folder_name)


##################################
# EXTRACT CONTOUR BY INDEX
##################################
def extract_contour_by_index(img, index=0):
    # Check if the image is already grayscale or has been loaded correctly
    if len(img.shape) == 2:  # Image is already grayscale
        gray = img
    elif len(img.shape) == 3:  # Image is BGR (color)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        print("Error: Invalid image format")
        return None
        
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
    

##################################
# PREPROCESS IMAGE FOR OCR
##################################
def preprocess_image_for_ocr(img, scale_factor=2, enhance=True):
    if len(img.shape) == 3:  # Convert to grayscale if the image has 3 channels (BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if enhance:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        img = clahe.apply(img)
    img_resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    _, img_thresh = cv2.threshold(img_resized, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_thresh

##################################
# EXTRACT BLUEPRINT TYPE
##################################
def extract_blueprint_type(image_path):
    img = cv2.imread(image_path)
    
    # Define margins or dimensions to crop to ROI where labels are located
    crop_height = 190
    crop_width = 280
    y_start = img.shape[0] - crop_height
    x_start = img.shape[1] - crop_width
    sidebar_corner = img[y_start:, x_start:]

    # Preprocess the image to enhance feature detection
    sidebar_corner = preprocess_image_for_ocr(sidebar_corner, enhance=True)
    
    # Extract the contour with an appropriate index
    contour = extract_contour_by_index(sidebar_corner, index=0)
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        contour_image = sidebar_corner[y:y+h, x:x+w]
        extracted_text = pytesseract.image_to_string(contour_image, config='--oem 3 --psm 6')
    else:
        extracted_text = "No contours detected"
        print(extracted_text)
        return None

    # Clean up the extracted text to remove unwanted characters
    cleaned_text = re.sub(r'[^A-Za-z0-9\-\.\s]', '', extracted_text)
    
    # Search for blueprint types in the cleaned extracted text
    pattern = r'([ASMP]-?1)\d{2}(?:\.\d+)?'  # Match the labels starting with "S-1", "A-1", "M-1", "P-1"
    matches = re.findall(pattern, cleaned_text)
    blueprint_types = []

    if matches:
        print("Blueprint Types Found:", matches)
        for match in matches:
            # Classify based on the main part of the label
            if match.startswith('S-1'):
                blueprint_types.append('Structural')
            elif match.startswith('A-1'):
                blueprint_types.append('Architectural')
            elif match.startswith('M-1'):
                blueprint_types.append('Mechanical')
            elif match.startswith('P-1'):
                blueprint_types.append('Plumbing')
        
        print("Classified Blueprint Types:", blueprint_types)
        return blueprint_types
    else:
        print("No blueprint type found in text.")
        return None


##################################
# EXTRACT TEXT FEATURES
##################################
def extract_text_features(image_path, blueprint_type):
    """
    Extract OCR text from the image, applying different cropping conditions based on the blueprint type.
    """
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Define cropping margins and scale factors for different blueprint types
    if blueprint_type == 'Architectural':
        top_margin, bottom_margin, left_margin, right_margin = 1100, 400, 150, 1600
        scale_factor = 1.  # Set specific scale factor for Architectural
    elif blueprint_type == 'Structural':
        top_margin, bottom_margin, left_margin, right_margin = 1100, 100, 500, 800
        scale_factor = 1.2  # Set specific scale factor for Structural
    else:  # Default cropping for Mechanical and Plumbing
        top_margin, bottom_margin, left_margin, right_margin = 1100, 50, 300, 700
        scale_factor = 1.2  # Set specific scale factor for Mechanical and Plumbing

    # Crop the image with the adjusted margins from all sides
    cropped_img = img[top_margin:height - bottom_margin, left_margin:width - right_margin]
    
    # Adjust the width using the specific scale factor for each blueprint type
    if scale_factor != 1:
        new_width = int(cropped_img.shape[1] * scale_factor)
        new_height = cropped_img.shape[0]  # Keep height the same
        cropped_img = cv2.resize(cropped_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale if the image is in color
    if len(cropped_img.shape) == 3:
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    # Extract text using OCR
    text = pytesseract.image_to_string(cropped_img)
    print(f"Extracted text:\n{text}")
    
    return text


##################################
# TEXT TO NUMBER CONVERSION
##################################
def text_to_number(text):
    # Mapping written numbers to their numeric counterparts
    mapping = {
        "FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4,
        "FIFTH": 5, "SIXTH": 6, "SEVENTH": 7, "EIGHTH": 8
    }

    # Check if text is None before attempting to call upper
    if text is None:
        return None
    return mapping.get(text.upper(), None)



##################################
# EXTRACT FLOOR NUMBER
##################################
def extract_floor_number(text):
    """
    Extracts numeric floor numbers, written floors (up to EIGHTH), 'ROOF PLAN', and 'CELLAR PLAN' from text.
    """
    # Preprocess the text for common OCR errors
    text = text.replace('Sth', '5th').replace('STH', '5TH').replace('Ast', '1st').replace('Ist', '1st')

    # Adjusted pattern to include optional characters and spaces
    pattern = r'''
        \b([1-9]|1[0-2])(?:st|nd|rd|th)?  # Numeric floors
        \s*(?:TO|--?|\s*-\s*|THRU)?\s*
        ([1-9]|1[0-2])?(?:st|nd|rd|th)?
        \s*[Ff][Ll][Oo][Oo][Rr]\s*[Pp][Ll][Aa][Nn]
        |
        \b(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH)
        (?:\s*(?:TO|--?|\s*-\s*|THRU)\s*(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH))?
        \s*[Ff][Ll][Oo][Oo][Rr]\s*[Pp][Ll][Aa][Nn]
        |
        \b(ROOF)\s*[Pp][Ll][Aa][Nn]
        |
        \b(CELL[AE]R)\s*[Ff]?[Ll]?[Oo]?[Oo]?[Rr]?[\s\-\.]*[Pp][Ll][Aa][Nn]
        |
        \b(FOUNDATION)\s*[Pp][Ll][Aa][Nn]
        |
        \b(SITE)\s*[Pp][Ll][Aa][Nn]
        |
        \b(PLOT)\s*[Pp][Ll][Aa][Nn]
    '''

    matches = re.finditer(pattern, text, re.IGNORECASE | re.VERBOSE | re.DOTALL)
    floor_set = set()

    for match in matches:
        if match.groups()[0]:
            # Numeric floor range
            start_floor, end_floor = match.groups()[0], match.groups()[1]
            if end_floor is None:
                end_floor = start_floor
            floor_set.update(str(i) for i in range(int(start_floor), int(end_floor) + 1))
        elif match.groups()[2]:
            # Written floor range
            start_floor, end_floor = text_to_number(match.groups()[2]), text_to_number(match.groups()[3])
            if end_floor is None:
                end_floor = start_floor
            floor_set.update(str(i) for i in range(start_floor, end_floor + 1))
        elif match.groups()[4]:
            # Roof plan
            floor_set.add('roof')
        elif match.groups()[5]:
            # Cellar plan
            floor_set.add('cellar')
        elif match.groups()[6]:
            # Foundation plan
            floor_set.add('foundation')
        elif match.groups()[7]:
            # Site plan
            floor_set.add('site')
        elif match.groups()[8]:
            # Plot plan
            floor_set.add('plot')

    if floor_set:
        return ', '.join(sorted(floor_set, key=lambda x: (not x.isnumeric(), x)))
    else:
        print("No floors found")
        return "N/A"


##################################
# EXTRACT SCALES
##################################
def extract_scales(text):
    # Adjusted pattern to match scales like 1/4"=1'-0", 1/4’=1'-0", and similar variations
    pattern = r'[Ss][Cc][Aa][Ll][Ee]\s*[:=]?\s*([\d\/]+[\'"°“”’]?\s*[=]?\s*\d+[\'"°“”’]?\s*-\s*\d+[\'"°“”’]?)'

    # Search for the pattern in the text
    match = re.search(pattern, text)

    if match:
        scale = match.group(1)

        # Clean up common OCR misreads and formatting
        corrections = {
            "V": "1", "l": "1", "I": "1", "7": "1", "f": "1", "°": "\"", "“": "\"", "”": "\"", "'": "\"", "’": "\"",
            "–": "-", "’": "'", "=": " = ", "T": "1"
        }

        # Replace misread characters
        for wrong, correct in corrections.items():
            scale = scale.replace(wrong, correct)

        # Ensure that 1/4 remains as 1/4 and does not get turned into 11/4
        scale = scale.replace("11/4", "1/4")

        # Remove any extra spaces around digits and units (e.g., 1 - 0" should be 1-0")
        scale = re.sub(r'\s*-\s*', '-', scale)  # Remove spaces around "-"
        scale = re.sub(r'\s*\/\s*', '/', scale)  # Remove spaces around "/"
        scale = re.sub(r'\s*\"\s*', '"', scale)  # Remove spaces around "
        scale = re.sub(r'\s*\=\s*', ' = ', scale)  # Ensure equal sign has spaces around it

        # Ensure that the scale ends correctly (e.g., add missing inches after dash if necessary)
        if scale.endswith("-"):
            scale = scale.rstrip("-") + "0\""

        # Normalize the format further if necessary (ensure format X" = Y"-Z")
        if "=" not in scale:
            parts = scale.split("-")
            if len(parts) == 2:
                scale = parts[0].strip() + "\" = " + parts[1].strip() + "\""

        return scale  # Return the corrected and formatted scale
    
    return "N/A"


##################################
# EXTRACT FLOOR ELEVATION
##################################
def extract_floor_elevation(text):
    """
    Extracts floor elevations from the text and associates them with their respective floor numbers.
    Handles both single and multiple floor elevations by using separate internal functions.
    """

    def extract_single_floor_elevation(text):
        """Extracts a single floor elevation and returns just the elevation value."""
        # Pattern for extracting a single floor elevation
        single_pattern = r'[Ff][Ll][Rr]\.?\s*[Ee][Ll]\.?\s*([\d,]+(?:\.\d+)?)\s*[\'"“”°]?'
        
        # Find a match for the single elevation
        match = re.search(single_pattern, text)
        
        if match:
            elevation = match.group(1).replace(',', '.').strip()
            if not elevation.endswith("'"):
                elevation += "'"
            return elevation
        return None

    def extract_multiple_floor_elevations(text):
        """Extracts multiple floor elevations and returns them as a formatted string."""
        # Pattern for extracting multiple floor elevations (with floor number and elevation)
        multi_pattern = r'(\d+)(?:ST|ND|RD|TH)?\s*[Ff][Ll][Rr]\.?\s*[Ee][Ll]\.?\s*([\d,]+(?:\.\d+)?)\s*[\'"“”°]?'
        
        # Find all matches for multiple elevations
        matches = re.findall(multi_pattern, text)
        
        if matches:
            elevations = []
            for match in matches:
                floor_number = match[0]  # Extract the floor number (e.g., 2, 3, etc.)
                elevation = match[1].replace(',', '.').strip()  # Normalize the decimal and remove extra commas
                if not elevation.endswith("'"):
                    elevation += "'"
                # Store the floor number and its corresponding elevation
                elevations.append(f"{floor_number}: {elevation}")
            
            # Return the formatted string for multiple elevations
            if elevations:
                return ', '.join(elevations)
        return None

    # Check if "FOUNDATION PLAN" is in the text
    if re.search(r'\bFOUNDATION\s*[Pp][Ll][Aa][Nn]', text, re.IGNORECASE):
        return "0'"

    # First, try extracting multiple floor elevations
    multi_floor_elevations = extract_multiple_floor_elevations(text)
    
    if multi_floor_elevations:
        return multi_floor_elevations

    # If no multiple elevations, try extracting a single elevation
    single_floor_elevation = extract_single_floor_elevation(text)
    
    if single_floor_elevation:
        return single_floor_elevation

    # If no elevations are found at all, return "N/A"
    return "N/A"

##################################
# APPLY BOUNDARY DETECTION
##################################
def apply_boundary_detection(image_path):
    """
    Detects and crops an image based on the largest and second-largest contours.
    Returns the final cropped image after drawing the outer boundary of the second-largest contour.
    """
    # Read the image
    img = cv2.imread(image_path)

    # Step 1: Extract the largest contour (index=0)
    largest_contour = extract_contour_by_index(img, index=0)
    if largest_contour is None:
        print("No largest contour found. Returning the original image.")
        return img, None  # Return the original image and None for bounding box

    # Step 2: Get bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = img[y:y + h - 10, x:x + w - 10]

    # Step 3: Extract the second largest contour within the cropped image
    second_largest_contour = extract_contour_by_index(cropped_image, index=0)
    if second_largest_contour is None:
        print("No second largest contour found. Returning the cropped image.")
        return cropped_image, None  # Return cropped image if second contour is not found

    # Step 4: Get bounding rectangle for the second-largest contour
    x2, y2, w2, h2 = cv2.boundingRect(second_largest_contour)

    # Step 5: Draw the second-largest contour (outer boundary)
    output_image = img.copy()
    # Draw the bounding rectangle for the second-largest contour on the cropped image
    cv2.rectangle(cropped_image, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

    # Step 6: Perform the final crop based on the second-largest contour
    final_cropped_image = cv2.cvtColor(cropped_image[y2:y2 + h2, x2:x2 + w2], cv2.COLOR_BGR2RGB)

    # Return the final cropped image and bounding box for further processing
    return final_cropped_image, (x2, y2, w2, h2)


##################################
# COMPUTE CORNER POINTS
##################################
def compute_corner_points(bounding_box):
    """
    Compute and format corner points based on the bounding box.
    """
    if bounding_box is None or len(bounding_box) != 4:
        print("Error: bounding_box does not contain the expected 4 values.")
        return "N/A"  # If no bounding box or invalid bounding box is provided

    # Unpack the bounding box
    try:
        x2, y2, w2, h2 = bounding_box
    except ValueError:
        print(f"Error: Expected 4 values for bounding_box, but got {len(bounding_box)}: {bounding_box}")
        return "N/A"

    # Calculate the four corners
    corners = [(x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2)]

    # Format the corners as strings in the desired format
    formatted_corners = [f"({corner[0]}, {corner[1]})" for corner in corners]

    return formatted_corners




