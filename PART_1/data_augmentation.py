#Data augmentation script. As of May 31st, the script is functional but has not been run for the presented data, and may be modified for parts 2 and 3 for the project.
import os
from PIL import Image, ImageEnhance
import random
import pandas as pd
import glob

#Data importation section
base_dir = "data"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
train_data = pd.read_csv(os.path.join(base_dir, 'train_dataset.csv'))

#Ensure the output directory exists
augmented_dir = os.path.join(base_dir, 'augmentation')
os.makedirs(augmented_dir, exist_ok=True)

#Clear the augmented output directory before starting
for file in glob.glob(os.path.join(augmented_dir,"*")):
    os.remove(file)

#Augmentation parameters
augmentation_factor = 2 #Number of augmented images to generate per source image

#Function to augment images
def augment_image(image_path, output_folder, combined_train_data, num_augmented_images=augmentation_factor, target_size=(48, 48)):
    img = Image.open(image_path)
    img_extension = os.path.splitext(image_path)[1]  #Extract file extension

    #Add original image to DataFrame
    extracted_label = train_data[train_data['Path'] == image_path]['Label'].values[0]
    if extracted_label in ['angry', 'happy', 'neutral', 'engaged']:
        label = extracted_label
        augmented_data = {'Path': image_path, 'Label': label, 'Name': os.path.basename(image_path)}
        combined_train_data.append(augmented_data)

        augmentations = [ # List of desired augmentation methods (image manipulation)
            ("rot", lambda img, degree: img.rotate(degree)),
            ("flip", lambda img, degree: img.transpose(Image.FLIP_LEFT_RIGHT)),
            ("col", lambda img, degree: ImageEnhance.Color(img).enhance(degree)),
            ("con", lambda img, degree: ImageEnhance.Contrast(img).enhance(degree)),
            ("bri", lambda img, degree: ImageEnhance.Brightness(img).enhance(degree)),
            ("sha", lambda img, degree: ImageEnhance.Sharpness(img).enhance(degree)),
            ("no", lambda img, degree: img),
        ]

        combinations = []
        while len(combinations) < num_augmented_images: #Generate random combinations of the possible augmentation methods
            combo = random.sample(augmentations, random.randint(1, len(augmentations)))
            combo_names = [name for name, _ in combo]
            if combo_names not in combinations:
                combinations.append(combo_names)

        for combo in combinations:
            augmented_img = img
            aug_name_parts = []
            for name in combo:
                aug_func = next(aug for aug_name, aug in augmentations if aug_name == name)
                if name == "rot":
                    degree = random.randint(-30, 30)
                elif name == "flip":
                    degree = 0
                else:
                    degree = round(random.uniform(0.5, 1.5), 2)

                augmented_img = aug_func(augmented_img, degree)
                aug_name_parts.append(f"{name}{degree}")

            aug_name = f"{'_'.join(aug_name_parts)}_{os.path.basename(image_path)}"
            aug_name = aug_name.replace(".", "_") + img_extension  #Add file extension
            aug_path = os.path.join(output_folder, aug_name)
            augmented_img.save(aug_path)
            augmented_data = {'Path': aug_path, 'Label': label, 'Name': aug_name}
            combined_train_data.append(augmented_data)
    else:
        print(f"Label not in ['angry', 'happy', 'neutral', 'engaged']: {extracted_label}")

    return combined_train_data

#Initialize combined_train_data
combined_train_data = []

#Augment training data
for index, row in train_data.iterrows():
    combined_train_data = augment_image(row['Path'], augmented_dir, combined_train_data)

#Write combined data to CSV
csv_file = os.path.join(base_dir, 'augmented_images.csv') #Includes original data
combined_train_df = pd.DataFrame(combined_train_data)
combined_train_df.to_csv(csv_file, index=False)

print(f'CSV file "{csv_file}" has been created with augmented data.')
