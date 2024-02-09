# Local Interaction Score

This Python script calculates and saves the local interaction scores from ColabFold-derived Predicted Aligned Error (PAE) result images instead of PAE value-containing json files. It processes PAE result images to evaluate the level of interaction in different quadrants of each image, based on the distribution of the blue color channel. The scores are then exported into an Excel file.

## Features

- Extraction of the blue channel from PAE maps generated from ColabFold.
- Image cropping to isolate Rank_1, Rank_2, Rank_3, Rank_4, and Rank_5 regions from PAE map.
- Calculation of interaction scores based on blue color distribution in different quadrants of the image.
- Exportation of interaction scores to an Excel file, with individual sheets for summary, grouped, and total results.

## Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/flyark/local-interaction-score.git
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
- Python 3.x
- OpenCV
- NumPy
- pandas
- Pillow

## Under the Hood
The script operates by first cropping images to isolate certain regions. The cropping is done using a fixed logic, but this can be modified based on specific use case needs.

Next, the script processes these images, dividing them into quadrants and extracting the blue channel from the image data. It then evaluates the distribution of this blue channel in each quadrant, calculating a 'blue average' and 'blue area' for each quadrant.

The interaction score is then computed based on these quadrant-level metrics, providing an evaluation of the interaction level in the image.

The script uses multi-threading to speed up the image processing tasks. All the calculated interaction scores are finally exported to an Excel file for further analysis.

## Author
Ah-Ram Kim (in Norbert Perrimon Lab at Harvard Medical School)
