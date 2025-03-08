
### imports 
import os 
import numpy as np 
import pandas as pd
import SimpleITK as sitk
from collections import defaultdict
import random




def rotate_mri_image(image, angle_degrees=10, axis='axial'):
    # Load the image
    size = image.GetSize()

    # Define front corner coordinates for averaging
    corners = [
        (0, 0, 0),  # Top-left-front
        (size[0] - 1, 0, 0),  # Top-right-front
        (0, size[1] - 1, 0),  # Bottom-left-front
        (size[0] - 1, size[1] - 1, 0)  # Bottom-right-front
    ]

    # Get pixel values at each front corner and calculate their average
    corner_values = [image[corner] for corner in corners]
    default_pixel_value = np.mean(corner_values)

    # Convert angle to radians
    angle_radians = np.deg2rad(angle_degrees)

    # Set up 3D rotation transformation
    transform = sitk.Euler3DTransform()
    rotation_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(size) / 2.0))
    transform.SetCenter(rotation_center)
    
    # Apply rotation based on axis
    if axis == 'axial':
        transform.SetRotation(0, 0, angle_radians)  # Rotate around Z-axis
    elif axis == 'coronal':
        transform.SetRotation(0, angle_radians, 0)  # Rotate around Y-axis
    elif axis == 'sagittal':
        transform.SetRotation(angle_radians, 0, 0)  # Rotate around X-axis
    else:
        raise ValueError("Axis must be 'axial', 'coronal', or 'sagittal'")

    # Apply the rotation with the default pixel value from the corners
    rotated_image = sitk.Resample(
        image,
        image,
        transform,
        sitk.sitkLinear,
        default_pixel_value,  # Use average of corner values as default
        image.GetPixelID()
    )

    return rotated_image

def shift_image_voxels(image, shift_voxels=(-5, -5, -5)):
    """
    Shifts a 3D image by a specified number of voxels.

    Parameters:
    - image_path: str, path to the 3D image file.
    - shift_voxels: tuple of int, number of voxels to shift in each (x, y, z) direction.

    Returns:
    - shifted_image: SimpleITK.Image, the shifted image.
    """
    # Load the image
    voxel_spacing = image.GetSpacing()
    size = image.GetSize()


    # Define front corner coordinates for averaging
    corners = [
        (0, 0, 0),  # Top-left-front
        (size[0] - 1, 0, 0),  # Top-right-front
        (0, size[1] - 1, 0),  # Bottom-left-front
        (size[0] - 1, size[1] - 1, 0)  # Bottom-right-front
    ]

    # Get pixel values at each front corner and calculate their average
    corner_values = [image[corner] for corner in corners]
    default_pixel_value = np.mean(corner_values)

    # Convert voxel shift to physical space shift (mm)
    shift_physical = [shift * spacing for shift, spacing in zip(shift_voxels, voxel_spacing)]

    # Set up the translation transform with the calculated physical shift
    transform = sitk.TranslationTransform(3)
    transform.SetOffset(shift_physical)

    # Apply the translation to the image
    shifted_image = sitk.Resample(
        image,
        image,
        transform,
        sitk.sitkLinear,
        default_pixel_value,  # Default value for new pixels
        image.GetPixelID()
    )

    return shifted_image

def aug(path):
    r1 = np.random.randint(-10,10)
    r2 = np.random.randint(0,10)

    img = sitk.ReadImage(path)
    img_shifted = shift_image_voxels(img,shift_voxels=(r1,r1,r1))
    img_rotated = rotate_mri_image(img_shifted,angle_degrees=r2)
    return img_rotated

def augment_dataset_to_balance(df, target_function, max_count,output):
    paths = df['preprocessed_path']
    ages = df['age']
    dataset = df['dataset']
    print(f'max count {max_count}')
    age_to_paths = defaultdict(list)
    for path, age, data in zip(paths, ages, dataset):
        age_to_paths[age].append((path, data))
    
    balanced_file_paths = []
    balanced_ages = []
    balanced_datasets = []
    
    for age, items in age_to_paths.items():
        for path, data in items:
            balanced_file_paths.append(path)
            balanced_ages.append(age)
            balanced_datasets.append(data)
        
        current_count = len(items)
        num_augmentations_needed = max(0, max_count - current_count)
        
        # print(f'for age {age} >>> number of augmentations needed: {num_augmentations_needed}')
        c = 0
        for z in range(num_augmentations_needed):
            path_to_augment, data = random.choice(items)  
            augmented_image = target_function(path_to_augment)  
            name = path_to_augment.split('5- normalized/')[1]

            os.makedirs(f'augmented_{output}', exist_ok=True)
            augmented_path = os.path.abspath(os.path.join(f'augmented_{output}', f'{age}_aug_{c}_{name}')) 
            sitk.WriteImage(augmented_image, augmented_path)  
            c += 1
            balanced_file_paths.append(augmented_path)
            balanced_ages.append(age)
            balanced_datasets.append(data)

    augmented_df = pd.DataFrame({
        'preprocessed_path': balanced_file_paths,
        'age': balanced_ages,
        'dataset': balanced_datasets,
        'aug': 1
    })
    df['aug'] = 0
    
    final = pd.concat([df, augmented_df]).reset_index(drop=True)
    final = final.drop_duplicates('preprocessed_path')
    out_path = os.path.join(f'augmented_{output}',f'{output}.csv')
    final.to_csv(out_path, index=False)
    return final 