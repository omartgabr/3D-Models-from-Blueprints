'''
Image Processing Pipeline

F24 DSE Capstone - NeARabl Inc
Author: Omar Gabr
Mentors: Jin Chen & Fani Maksakauli

This file generates a program that takes in a raw 2D blueprint image,
performs a series of image processing operations to achieve
a number of tasks including:
(1) floorplan labeling
(2) outer wall detection
(3) measurement scale recognition

'''

# custom functions definitions standalone file
import ImageProcessing_functions as IPF

raw_images_folder_path = 'C:/Users/Omar G/Desktop/CCNY DSE/1.DSE I9800 Capstone Project/raw data'     # folder that contains raw images
output_folder_path = 'C:/Users/Omar G/Desktop/CCNY DSE/1.DSE I9800 Capstone Project/Labeled_Floorplans/'     # new folder that will contain subfolders of labeled images

templates = {
    'Mechanical': 'C:/Users/Omar G/Desktop/CCNY DSE/1.DSE I9800 Capstone Project/3D-Models-from-Blueprints/Floorplan labels/Mech-general-label.png',
    'Plumbing': 'C:/Users/Omar G/Desktop/CCNY DSE/1.DSE I9800 Capstone Project/3D-Models-from-Blueprints/Floorplan labels/Plumb-general-label.png',
    'Architectural': 'C:/Users/Omar G/Desktop/CCNY DSE/1.DSE I9800 Capstone Project/3D-Models-from-Blueprints/Floorplan labels/Arch-general-label.png',
    'Structural': 'C:/Users/Omar G/Desktop/CCNY DSE/1.DSE I9800 Capstone Project/3D-Models-from-Blueprints/Floorplan labels/Struct-label.png',
}

# clear files in folders for proper reset
IPF.delete_files_in_folder(output_folder_path)

# 1st Step: Floorplan Labeling
# create dictionary that stores floorplan label and page number
metadata = IPF.extract_floorplan_label(raw_images_folder_path, templates)

# 2nd Step: Outside-of-Boundary Template Matching
# detection of additional features like floor number, elevation, scales


# 3rd Step: Edge Boundary Detection
# store additional information like corners of rectangular contour
metadata = IPF.outer_edge_detection(raw_images_folder_path, output_folder_path, metadata)

# Final Step
# store images to respective subfolders using metadata
# modify function to NOT create json for uncategorized images
IPF.create_json_files(metadata, output_folder_path)




