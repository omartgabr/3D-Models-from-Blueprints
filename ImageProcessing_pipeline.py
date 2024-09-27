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

########################### FLOORPLAN CLASSIFICATION ###########################

raw_images_folder_path = '../raw data/'     # folder that contains raw images
output_folder_path = '../'     # new folder that will contain subfolders of labeled images

templates = {
    'Mechanical': '/3D-Models-from-Blueprints/Floorplan templates/Arch-general-label.png',
    'Plumbing': '/3D-Models-from-Blueprints/Floorplan templates/Plumb-general-label.png',
    'Architectural': '/3D-Models-from-Blueprints/Floorplan templates/Arch-general-label.png',
    'Structural': '/3D-Models-from-Blueprints/Floorplan templates/Struct-label.png',
}

# execute function to process and classify images
IPF.extract_floorplan_label(raw_images_folder_path, templates, output_folder_path)


########################### PREPROCESSING OF FLOORPLAN TYPES ###########################
############## ARCHITECTURAL ##############



