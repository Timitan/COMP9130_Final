import kagglehub
import os
import shutil

# Download latest version
os.makedirs("./data", exist_ok=True)
path = kagglehub.dataset_download("shuvoalok/dawn-dataset", output_dir="./data")

print("data/", path)

#make paths for filtered data
output_dir = "./data/filtered"
os.makedirs(os.path.join(output_dir, "snow"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "rain"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "fog"), exist_ok=True)

#filter data into snow, rain and fog folders
def filter_data(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            if "snow" in filename:
                shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, "snow", filename))
            elif "rain" in filename:
                shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, "rain", filename))
            elif "fog" in filename:
                shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, "fog", filename))
            elif "mist" in filename:
                shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, "rain", filename))
            elif "haze" in filename:
                shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, "fog", filename))

# Filter the data
filter_data("./data/images", output_dir)
        