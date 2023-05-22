import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import shutil
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import sys



def crop_image(pae_dir, crop_dir, filename, rank):
    x1, y1 = (60 + ((rank - 1) * 481)), 64
    x2, y2 = (366 + ((rank - 1) * 481)), 370

    image_path = os.path.join(pae_dir, filename)
    crop_path = os.path.join(crop_dir, filename)

    if not os.path.exists(crop_path):
        image = Image.open(image_path)
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image.save(crop_path)

def parallel_crop(pae_dir):
    image_files = [file for file in os.listdir(pae_dir) if file.endswith('.png')]

    for rank in range(1, 6):
        crop_dir = os.path.join(pae_dir, f'croped_pae_rank_{rank}')
        if not os.path.exists(crop_dir):
            os.makedirs(crop_dir)

        with ThreadPoolExecutor() as executor:
            executor.map(lambda filename: crop_image(pae_dir, crop_dir, filename, rank), image_files)

def move_small_files_to_empty_folder(parent_folder, target_size=1246):
    subfolders = [folder for folder in os.listdir(parent_folder) if folder.startswith('croped_pae_rank_')]

    for subfolder in subfolders:
        subfolder_path = os.path.join(parent_folder, subfolder)
        empty_folder_path = os.path.join(subfolder_path, 'empty')

        # Create the 'empty' folder if it does not exist
        if not os.path.exists(empty_folder_path):
            os.mkdir(empty_folder_path)

        for image_file in os.listdir(subfolder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subfolder_path, image_file)
                file_size = os.path.getsize(image_path)

                if file_size == target_size:
                    shutil.move(image_path, os.path.join(empty_folder_path, image_file))


def is_valid_file_name(file_name):
    return file_name.count('___') == 1

def split_image(image_path):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    # Find x-axis and y-axis areas
    x_axis_area = []
    for y in range(height):
        if np.all(gray[y, :] < 50):  # Check if all pixels in the row are black (or nearly black)
            x_axis_area.append(y)

    y_axis_area = []
    for x in range(width):
        if np.all(gray[:, x] < 50):  # Check if all pixels in the column are black (or nearly black)
            y_axis_area.append(x)

    # print(image_path)

    # Divide the image into quadrants based on the x-axis and y-axis areas
    if x_axis_area is not None and y_axis_area is not None:
        q1 = image[0:x_axis_area[0], y_axis_area[-1] + 1:width]
        q2 = image[0:x_axis_area[0], 0:y_axis_area[0]]
        q3 = image[x_axis_area[-1] + 1:height, 0:y_axis_area[0]]
        q4 = image[x_axis_area[-1] + 1:height, y_axis_area[-1] + 1:width]


        if q1.size == 0 or q2.size == 0:  # Check if q1 is empty
            print(f"Empty quadrant 1 or 2: {image_path}")
            return image_path

        q1_blue = extract_blue_channel(q1)
        q2_blue = extract_blue_channel(q2)
        q3_blue = extract_blue_channel(q3)
        q4_blue = extract_blue_channel(q4)

    q1_blue_intensity = get_blue_intensity_average(q1)/255
    q2_blue_intensity = get_blue_intensity_average(q2)/255
    q3_blue_intensity = get_blue_intensity_average(q3)/255
    q4_blue_intensity = get_blue_intensity_average(q4)/255

    # print("Average blue intensity in Quadrant 1:", q1_blue_intensity)
    # print("Average blue intensity in Quadrant 2:", q2_blue_intensity)
    # print("Average blue intensity in Quadrant 3:", q3_blue_intensity)
    # print("Average blue intensity in Quadrant 4:", q4_blue_intensity)

    q1_blue_area = get_blue_area(q1)
    q2_blue_area = get_blue_area(q2)
    q3_blue_area = get_blue_area(q3)
    q4_blue_area = get_blue_area(q4)


    interaction_area = (q1_blue_area + q3_blue_area) / (q2_blue_area + q4_blue_area) * 100

    interaction_intensity = (q1_blue_intensity + q3_blue_intensity) / 2

    return q1_blue_intensity, q2_blue_intensity, q3_blue_intensity, q4_blue_intensity, q1_blue_area, q2_blue_area, q3_blue_area, q4_blue_area, interaction_area, interaction_intensity

def extract_blue_channel(image, threshold=325):
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize the image values to the range [0, 300]
    normalized_image = cv2.normalize(rgb_image, None, 0, threshold, cv2.NORM_MINMAX)

    # Extract the blue channel
    gray_image = normalized_image[:, :, 0]

    return gray_image

def get_blue_intensity_average(image):
    blue_channel = extract_blue_channel(image)
    inverted_blue_channel = 255 - blue_channel
    
    if np.isnan(inverted_blue_channel).any() or np.isinf(inverted_blue_channel).any():
        # Handle missing or invalid values
        raise ValueError("Invalid values in inverted_blue_channel array")
    
    non_zero_pixels = inverted_blue_channel[inverted_blue_channel != 0]
    
    if non_zero_pixels.size == 0:
        # Handle case when no non-zero pixels exist
        return 0.0  # Or return a default value
    
    average_non_zero = np.mean(non_zero_pixels)
    return average_non_zero

def get_blue_area(image):
    blue_channel = extract_blue_channel(image)
    inverted_blue_channel = 255 - blue_channel
    inverted_blue_channel[inverted_blue_channel != 0] = 1
    blue_area = np.mean(inverted_blue_channel)
    return blue_area * 100

def process_images(parent_folder):
    # Initialize an empty list to store the results
    results = []

    empty_quadrant_data = []

    # List all subfolders in the parent folder
    subfolders = [folder for folder in os.listdir(parent_folder) if folder.startswith('croped_pae_rank_')]

    # Process each subfolder
    for subfolder in subfolders:
        # Get the full path to the subfolder
        subfolder_path = os.path.join(parent_folder, subfolder)

        # List all files in the subfolder
        all_files = os.listdir(subfolder_path)

        # Filter out the image files (assuming they have .jpg, .png, or .jpeg extensions)
        image_files = [file for file in all_files if file.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # Process each image file
        for image_file in image_files:
            if not is_valid_file_name(image_file):
                continue

            image_path = os.path.join(subfolder_path, image_file)
            result = split_image(image_path)
            if result == image_path:
                print("Empty quadrant 1:", result)
                empty_quadrant_data.append({'image_path': result})
                continue

            q1_blue_intensity, q2_blue_intensity, q3_blue_intensity, q4_blue_intensity, q1_blue_area, q2_blue_area, q3_blue_area, q4_blue_area, interaction_area, interaction_intensity = result

            # Get the rank from the subfolder name
            rank = int(subfolder.split('_')[-1])

            # Append the results to the list
            results.append({
                'pae_file_name': image_file,
                'rank': rank,
                'q1_blue_average': q1_blue_intensity,
                'q2_blue_average': q2_blue_intensity,
                'q3_blue_average': q3_blue_intensity,
                'q4_blue_average': q4_blue_intensity, 
                'q1_blue_area': q1_blue_area, 
                'q2_blue_area': q2_blue_area, 
                'q3_blue_area': q3_blue_area, 
                'q4_blue_area': q4_blue_area, 
                'interaction_area': interaction_area, 
                'interaction_intensity': interaction_intensity
            })

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)
    
    # Create a DataFrame from the empty quadrant data list
    empty_quadrant_df = pd.DataFrame(empty_quadrant_data)
    
    return results_df, empty_quadrant_df

def move_small_files_to_empty_folder(parent_folder, target_size=1246):
    subfolders = [folder for folder in os.listdir(parent_folder) if folder.startswith('croped_pae_rank_')]

    for subfolder in subfolders:
        subfolder_path = os.path.join(parent_folder, subfolder)
        empty_folder_path = os.path.join(subfolder_path, 'empty')

        # Create the 'empty' folder if it does not exist
        if not os.path.exists(empty_folder_path):
            os.mkdir(empty_folder_path)

        for image_file in os.listdir(subfolder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subfolder_path, image_file)
                file_size = os.path.getsize(image_path)

                if file_size == target_size:
                    shutil.move(image_path, os.path.join(empty_folder_path, image_file))


def is_valid_file_name(file_name):
    return file_name.count('___') == 1

def main(input_folder, output_file):
    # Process the images and store the results in a DataFrame
    parallel_crop(input_folder)
    move_small_files_to_empty_folder(input_folder)

    # Get current date as a string in the format YYYYMMDD
    current_date = datetime.now().strftime('%Y%m%d')

    # Call the process_images function
    results_df, empty_quadrant_df = process_images(input_folder)

    # Save the DataFrame to a CSV file
    results_df.to_csv(f'blue_color_{current_date}.csv', index=False)

    # Reload the DataFrame from the CSV file
    results_df_single = pd.read_csv(f'blue_color_{current_date}.csv')

    # Sort DataFrame based on 'pae_file_name' ascending and 'interaction_area' descending
    sorted_df = results_df_single.sort_values(by=['pae_file_name', 'interaction_area'], ascending=[True, False])

    # reset_index() to make groupby-derived dataframe into a standard dataframe.
    grouped_df = sorted_df.groupby('pae_file_name').mean().reset_index()

    # Add '_avg' to column names except for 'pae_file_name'
    grouped_df.columns = ['pae_file_name'] + [col + '_avg' for col in grouped_df.columns if col != 'pae_file_name']

    # Sort DataFrame based on 'pae_file_name' ascending and 'interaction_area' descending
    sorted_df = results_df_single.sort_values(by=['pae_file_name', 'interaction_area'], ascending=[True, False])

    # Get the top value from each 'pae_file_name' group based on 'interaction_area'
    top_values = sorted_df.groupby('pae_file_name')['interaction_area'].idxmax()
    top_df = sorted_df.loc[top_values]

    merged_df = pd.merge(top_df, grouped_df, on='pae_file_name')
    merged_df = merged_df.drop(columns=['rank', 'rank_avg'])

    # Write the results_df_rank1 and grouped_df dataframes to separate sheets in an Excel file
    with pd.ExcelWriter(output_file) as writer:
        merged_df.to_excel(writer, sheet_name='summary')
        grouped_df.to_excel(writer, sheet_name='Grouped')
        sorted_df.sort_values(["pae_file_name", "rank"]).to_excel(writer, sheet_name='Total')

    print(f'Excel file saved as: {output_file}')


if __name__ == '__main__':
    # Check if the required arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python function_name.py <input_folder> <output_excel_file>")
        sys.exit(1)
    
    # Get the input folder and output Excel file paths from the command line arguments
    input_folder = sys.argv[1]
    output_excel_file = sys.argv[2]

    # Call the main function with the provided input and output paths
    main(input_folder, output_excel_file)
