# Local Interaction Score

This repository contains a script for calculating the local interaction score of images in a given folder. The local interaction score measures the interaction between different quadrants of an image based on blue color intensity and area.

## Features

- Image cropping: The script crops images into quadrants based on predefined coordinates.
- Blue color analysis: It calculates the average blue color intensity and area for each quadrant of the cropped images.
- Local interaction score: It computes the local interaction score based on the blue color intensity and area of the quadrants.
- CSV output: The results are saved in a CSV file with detailed information about each image.
- Excel output: The script generates an Excel file summarizing the local interaction scores and provides grouped and total analysis.

## Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/local-interaction-score.git
   ```
   
2. Navigate to the repository directory:

   ```shell
   cd local-interaction-score
   ```
3. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```
4. Execute the script with the desired input folder and output Excel file:

   ```shell
   python local_interaction_score.py /path/to/input/folder /path/to/output/excel.xlsx
   ```
   
Replace /path/to/input/folder with the actual path to the folder containing the input images, and replace /path/to/output/excel.xlsx with the desired path for the output Excel file.

Check the generated Excel file containing the local interaction scores and analysis.

## Requirements
Python 3.x
OpenCV
NumPy
pandas
Pillow

