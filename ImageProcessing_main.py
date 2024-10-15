'''
Image Processing Pipeline

F24 DSE Capstone - NeARabl Inc
Author: Omar Gabr
Mentors: Jin Chen & Fani Maksakauli

This file generates a program that takes in raw 2D blueprint images,
performs a series of image processing operations to achieve
a number of tasks including:

(1) Blueprint type extraction
(2) Floor level extraction
(3) Scales measurement extraction
(4) Floor elevation extraction
(5) Outer Wall Layout detection
(6) Boundary corners extraction

'''

# custom functions definitions standalone file
import ImageProcessing_functions as IPF
import os
import json
import cv2


# raw data folder
raw_images_folder = 'raw data/'

# create folder to store the processed data
folder_name = 'processed_data'
output_path = IPF.create_processed_data_folder(folder_name)

# initialize blueprint type list
blueprint_type = ['Architectural', 'Structural', 'Mechanical', 'Plumbing']

# Loop through each image file in the folder
for image_file in os.listdir(raw_images_folder):
    image_path = os.path.join(raw_images_folder, image_file)
    if os.path.isfile(image_path):
        print(f"\nProcessing image: {image_file}")

        # extract blueprint type from OCR text
        blueprint_type = IPF.extract_blueprint_type(image_path)

        # Skip the image if blueprint type was not classified
        if not blueprint_type:
            print(f"Image {image_file} was not classified. Skipping further operations.")
            continue  # Skip to the next image
        
        # extract page number from the filename
        page_num = int(image_file.split('_')[1].split('.')[0])

        # extract OCR text from images then extract separate features
        # extract floor number from OCR text
        ocr_text = IPF.extract_text_features(image_path, blueprint_type)
        floor_number = IPF.extract_floor_number(ocr_text)

        # extract scales from OCR text
        scales = IPF.extract_scales(ocr_text)
        print(f"Scales: {scales}")

        # extract elevation from OCR text -- this is only found in structural blueprint types
        elevation = IPF.extract_floor_elevation(ocr_text)
        print(f"Floor Elevation: {elevation}") 

        # compute outer edge detection and extract boundary corners
        detected_layout, bbox = IPF.apply_boundary_detection(image_path)
        detected_corners = IPF.compute_corner_points(bbox)
        print(f"Boundary Corners: {detected_corners}")
    
        # construct metadata dictionary to convert to json
        metadata = {
            "blueprint type": blueprint_type,
            "scale": scales,
            "floor": floor_number,
            "elevation": elevation,
            "boundary corners": detected_corners,  # Format as needed
            "page number": page_num
        }

        # create filename and paths for json + images
        json_filename = f"{os.path.splitext(image_file)[0]}_json.json"
        json_path = os.path.join(output_path, json_filename)

        detected_layout_filename = f"{os.path.splitext(image_file)[0]}.png"
        detected_layout_path = os.path.join(output_path, detected_layout_filename)

        cv2.imwrite(detected_layout_path, detected_layout)

        # write the processed images with their corresponding json files into the output path
        with open(json_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)  # Save with indentation for readability






