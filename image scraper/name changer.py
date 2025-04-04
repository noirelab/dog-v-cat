import os

def folder_name_changing(base_dir, suffix):
    # Iterate over each item in the directory
    for folder in os.listdir(base_dir):
        # Construct the full path
        folder_path = os.path.join(base_dir, folder)
        # Check if it is a directory and ends with " dog"
        if os.path.isdir(folder_path) and folder.endswith(suffix):
            # Remove the " dog" suffix (which is 4 characters: a space and "dog")
            new_name = folder[:-4]
            new_folder_path = os.path.join(base_dir, new_name)
            # Rename the folder
            os.rename(folder_path, new_folder_path)
            print(f'Renamed "{folder}" to "{new_name}"')

def file_name_changing(base_dir, prefix):
    # Define image extensions you want to process
    image_extensions = ('.png', '.jpg', '.jpeg', 'webp')

    # Iterate over each file in the directory
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)
        # Check if it's a file and its extension matches an image type
        if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
            # Prevent adding the prefix multiple times
            if not filename.startswith(prefix):
                new_filename = prefix + filename
                new_file_path = os.path.join(base_dir, new_filename)
                os.rename(file_path, new_file_path)
                print(f'Renamed "{filename}" to "{new_filename}"')


# folder_name_changing
# base_dir = "D:/Google-ImageScraper-/photos"
# suffix = " dog"
# folder_name_changing(base_dir, suffix)

# file_name_changing
base_dir = 'D:/Google-Image-Scraper/backup/Yorkshire_terrier'
prefix = 'n02094433_'
file_name_changing(base_dir, prefix)
