
Script takes 2 arguments: source_folder and output_folder, and locates the page from an image, straight it and save it to new image.
The script simulates the CamScanner App. 
Assumptions:
1. The page is the main object in the image.
2. The page is rectangle.
3. The height is bigger than width.

Package needs to be install before using: 
opencv-contrib-python==4.5.4.58

Then write in your command line:
python hw2.py source_folder output_folder

source_folder : path to directory which contains the images (use this pattern: "[insert_path_here]")
output_folder : path to directory which contains the converted images(use this pattern: "[insert_path_here]")