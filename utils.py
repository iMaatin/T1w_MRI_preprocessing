### imports 

import os 
import numpy as np 
import pandas as pd
import ants 
from ipywidgets import interact
from helpers import * 
import glob 
import time
import time
from functools import wraps
import SimpleITK as sitk
from antspynet.utilities import brain_extraction
import subprocess
import itertools
from tqdm import tqdm
import datetime


mni = ants.image_read('mni_icbm152_t1_tal_nlin_sym_09a.nii',reorient='IAL')
mni_mask = ants.image_read('mni_icbm152_t1_tal_nlin_sym_09a_mask.nii',reorient='IAL')


### defining the logger as a decorator function 

results_df = pd.DataFrame(columns=["Function Name", "Execution Time (s)", "Output"])

def logger (func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        t1 = time.time()

        processed_img, processed_img_path = func(*args,**kwargs)

        t2 = time.time()
        exec_time = t2 - t1 

        global results_df
        results_df = pd.concat([results_df, pd.DataFrame({
            "Function Name": [func.__name__],
            "Execution Time (s)": [exec_time],
            "Output": [processed_img_path]
        })], ignore_index=True)
        return processed_img, processed_img_path
    
    return wrapper


### folder maker function
def mk_folders(file_url, custom_folder_name):
    # file_url = file_url.split('https://file+.vscode-resource.vscode-cdn.net')[1]
    p2 = os.path.basename(file_url)
    
    directory_path = os.path.dirname(file_url)
    first_folder_name = p2.split('.')[0]  # Extract the file name without the extension
    first_folder_path = os.path.join(directory_path, first_folder_name)
    
    os.makedirs(first_folder_path, exist_ok=True)
    
    second_folder_path = os.path.join(first_folder_path, custom_folder_name)
    os.makedirs(second_folder_path, exist_ok=True)
    
    return second_folder_path, p2 

### folder maker function
def mk_folders2(file_url, custom_folder_name):
    # file_url = file_url.split('https://file+.vscode-resource.vscode-cdn.net')[1]
    p2 = os.path.basename(file_url)
    
    directory_path = os.path.dirname(file_url)
    parent_directory_path = os.path.dirname(directory_path)

    first_folder_path = os.path.join(parent_directory_path, custom_folder_name)
    
    os.makedirs(first_folder_path, exist_ok=True)

    return first_folder_path, p2 

### step 1 
### reading the image and bias correction using sitk library 
@logger
def bias_correction(path:str, shrink_factor:int = 4,plot=False):
    # path = path.split('https://file+.vscode-resource.vscode-cdn.net')[1]
    # print(path)
    img = sitk.ReadImage(path, sitk.sitkFloat32)
    img = sitk.DICOMOrient(img,'RPS')
    rescaled = sitk.RescaleIntensity(img, 0, 255)
    head = sitk.LiThreshold(rescaled,0,1)
    shrinked_img = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
    shrinked_head = sitk.Shrink(head, [shrink_factor] * img.GetDimension())
    corr = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corr.Execute(shrinked_img,shrinked_head)
    log_bias_field = corr.GetLogBiasFieldAsImage(img)
    full_res_img = img / sitk.Exp(log_bias_field)
    if plot == True:
        explore_3D_array_comparison(sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(full_res_img),cmap='viridis')
    else:
        pass

    processed_folder, pt = mk_folders(path, '1- bias_corrected')
    # filename = os.path.basename(path)
    # folder = os.path.join(os.path.dirname(path), filename.split('.')[0])
    # os.makedirs(folder, exist_ok=True)
    
    outpath = os.path.join(processed_folder, pt)
    # print(outpath)
    # Write the corrected image to the output path
    sitk.WriteImage(full_res_img, outpath)
    return full_res_img, outpath


### step 2 
## non linear registration with antspy library 
# mainly using Syn algorithm    

@logger
def registration(path: str, template, alg:str, plot = False ):
    img = ants.image_read(path,reorient='IAL')
    reg = ants.registration(fixed=template, moving=img, type_of_transform=alg, verbose=False)
    reg_img = reg['warpedmovout']
    # resampled_img = ants.resample_image(reg_img, img.shape, use_voxels=True) ## just in case we need it later
    if plot == True:
        explore_3D_array(reg_img.numpy(), cmap='viridis')
    else: 
        pass 

    processed_folder, pt = mk_folders2(path, '2- registered')
    outpath = os.path.join(processed_folder, pt)
    reg_img.to_file(outpath)
    return reg_img, outpath


### step 3 brain extraction using CNN method from anstpynet library 

@logger
def brainer(path:str):
    img = ants.image_read(path,reorient='IAL')
    prob_brain_mask = brain_extraction(img, modality='t1',verbose=False)
    brain_mask = ants.get_mask(prob_brain_mask, low_thresh=0.5)
    skull_stripped = ants.mask_image(img, brain_mask)
    processed_folder, pt = mk_folders2(path, '3- brain_extracted')
    outpath = os.path.join(processed_folder, pt)
    ptt = pt.split('.nii.gz')[0]
    mask_outpath = os.path.join(processed_folder, ptt + '_mask.nii.gz')

    brain_mask.to_file(mask_outpath)
    skull_stripped.to_file(outpath)

    return skull_stripped, outpath

### step 4 
### added resampling to have 160*192*160 size needed for the model 
@logger
def resampling (path:str, interploator:int = 4):
    '''
    interp_type : integer
    one of 0 (linear), 1 (nearest neighbor), 2 (gaussian), 3 (windowed sinc), 4 (bspline)
    '''
    img = ants.image_read(path,reorient='IAL')
    resampled = ants.resample_image(img, (160,192,160), use_voxels=True,interp_type=interploator)
    processed_folder, pt = mk_folders2(path, '4- resampled')
    outpath = os.path.join(processed_folder, pt)
    resampled.to_file(outpath)
    return resampled, outpath

### step 5
## normalization: mean is set to 0 and sd is set to 1 

@logger 
def normalizer (path:str):
    img = sitk.ReadImage(path, sitk.sitkFloat32)
    img = sitk.DICOMOrient(img,'RPS')
    norm_img = sitk.Normalize(img)
    processed_folder, pt = mk_folders2(path, '5- normalized')
    
    def norm_check(raw):
        arr = sitk.GetArrayFromImage(raw)
        arr.reshape(-1)
        mean = arr.mean().round(2)
        sd = arr.std().round(2)
        print (f'{mean} and {sd}')

    # norm_check(norm_img)
    # explore_3D_array_comparison(sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(norm_img),cmap='viridis')
    outpath = os.path.join(processed_folder, pt)
    sitk.WriteImage(norm_img, outpath)
    return norm_img, outpath

### defining a function that does all the steps above consecutively 


def aio (path:str, reg_template, ip, bias_shrink_factor:int = 4, 
         reg_alg:str ='SyN', plot=False):
    i1, p1 = bias_correction(path,shrink_factor=bias_shrink_factor)
    i2, p2 = registration(p1,template=reg_template,alg=reg_alg)
    i3, p3 = brainer(p2)
    i4, p4 = resampling(p3, interploator=ip)
    i5, p5 = normalizer(p4)
    

    if plot == True:
        explore_3D_array(sitk.GetArrayFromImage(i5),cmap='viridis')
    else: 
        pass 

    return i5, p5, [p1,p2,p3,p4] 
   
def delete_paths(paths):
    """
    Deletes files or directories provided in the paths list.

    Parameters:
    paths (list): List of paths (files/directories) to be deleted.
    """
    for path in paths:
        try:
            if os.path.isdir(path):
                os.rmdir(path)  # Remove the directory if it's empty
            elif os.path.isfile(path):
                os.remove(path)  # Remove the file
            # print(f"Deleted: {path}")
        except Exception as e:
            print(f"Failed to delete {path}: {e}")

def apply_aio(df_path: str, convert_to_nifti: bool, clean: bool, save: bool, reg_template: str=mni, reg_alg='SyN'):
    '''
    Arguments:
    
    df_path is a string that points to the CSV file containing a dataframe. The dataframe has a 'path' column with the address of the NIfTI or DICOM files.
    If the images are already converted to NIfTI files, set `convert_to_nifti` to False; otherwise, the function assumes that the given path is a NIfTI file address.
    '''

    def convert_dicom_to_nifti(dicom_folder_path, img_id):
        """
        Convert a folder of DICOM files to NIfTI format using dcm2niix.

        Parameters:
        dicom_folder_path (str): Path to the folder containing DICOM files.
        """
        # Ensure dcm2niix is available in the system PATH
        try:
            subprocess.run(['dcm2niix', '-h'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise EnvironmentError("dcm2niix is not installed or not found in PATH.")

        # Run dcm2niix command
        command = [
            'dcm2niix',  # The dcm2niix executable
            '-z', 'y',   # Compress the output NIfTI file (gzip)
            '-f', img_id,  # Format for the output filename: series description and series number
            '-o', dicom_folder_path,  # Output directory
            dicom_folder_path  # Input DICOM folder
        ]
        
        # Execute the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e.stderr.decode()}")
            raise
        out = os.path.join(dicom_folder_path, f'{img_id}.nii.gz')
        return out 

    # Load the dataframe
    now = datetime.datetime.now()
    time = now.strftime(r'%Y%m%d_%H%M%S')
    df = pd.read_csv(df_path)
    df_path_dir = os.path.dirname(df_path)
    processed_df_dir = os.path.join(df_path_dir,'processed')
    os.makedirs(processed_df_dir,exist_ok=True)
    out = os.path.join(processed_df_dir, f'processed_{time}.csv')
    l = len(df)
    p5_results = []
    invalid_paths = []  

    # Wrap the for loop with tqdm for a progress bar
    for path in tqdm(df['path'], total=l, desc="Processing Paths", unit="file"):
        # Check if the path exists
        if not os.path.exists(path):
            print(f"Skipping invalid path: {path}")
            invalid_paths.append(path)
            p5_results.append('path error')
            continue  # Skip this iteration if the path is invalid

        if convert_to_nifti:
            img_id = path.split('/ADNI/')[2].split('/')[3]
            path = convert_dicom_to_nifti(path, img_id=img_id)

        try:
            # Process the image using the `aio` function
            _, p5, inter_paths = aio(path=path, reg_template=reg_template, ip=4, bias_shrink_factor=4, reg_alg=reg_alg, plot=False)
            p5_results.append(p5)
            if clean:
                delete_paths(inter_paths)
            # inters.append(inter_paths)
        except(RuntimeError):
            p5_results.append('error')
            invalid_paths.append(path)

    # Add the preprocessed paths to the dataframe
    try:
        df['preprocessed_path'] = p5_results
        if save:
            df.to_csv(out, index=False)
    except ValueError:
        if save: 
            p5_ser = pd.Series(p5_results)
            bonked_out = os.path.join(df_path_dir,f'bonked_{time}.csv')
            p5_ser.to_csv(bonked_out,index=False)

    return df, invalid_paths

### bids specific functions 

def table_prep(root_path, drop:bool=True):

    tsv_path = os.path.join(root_path, 'participants.tsv')
    df = pd.read_csv(tsv_path, sep='\t')

    def df_path(participant, session=None):
        if session:
            return os.path.join(participant, session)
        return participant
    
    if 'session_id' in df.columns and df['session_id'].notna().any():
        df['folder'] = np.vectorize(df_path)(df['participant_id'], df['session_id'])
    else:
        df['folder'] = np.vectorize(df_path)(df['participant_id'])


    for index, i in df.iterrows():
        folder = i['folder']
        path = os.path.join(root_path, folder)
        imgs = glob.glob(f'{path}/**/*.nii.gz', recursive=True)

        if imgs:
            df.at[index, 'path'] = imgs[0]

        if len(imgs) > 1: 
            print(f'WARNING!! More than 1 image identified for {folder}')
    if drop:
        df = df.dropna()

    df_outpath = os.path.join(root_path, 'preprocess.csv')
    df.to_csv(df_outpath, index=False)

    return df

def chunker(root_path:str, chunk_size:int=200, file_prefix:str="chunk"):
    df_path = os.path.join(root_path, 'preprocess.csv')
    df = pd.read_csv(df_path)
    chunks = []
    
    if len(df) > 200:
        num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)

        for i in range(num_chunks):
            chunk = df[i * chunk_size: (i + 1) * chunk_size]
            filename = f"{file_prefix}_{i+1}_camcan.csv"
            chunk.to_csv(filename, index=False)
            chunks.append(chunk)
            print(f"Saved: {filename}")
    else:
        chunks.append(df_path)
    
    return chunks

def concat_csv_files(directory_path):
        csv_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]
        
        dataframes = [pd.read_csv(csv_file) for csv_file in csv_files]
        concatenated_df = pd.concat(dataframes, axis=0, ignore_index=True)
        
        return concatenated_df

def bids_aio (root_path:str):
    table_prep(root_path)
    chunks = chunker(root_path)

    for i in chunks:
        apply_aio(
            df_path=i, 
            convert_to_nifti=False,
            clean=True,
            save=True,
            reg_template=mni, 
        )
        os.remove(i)

    processed_folder = os.path.join(root_path, 'processed')
    final_processed_df = concat_csv_files(processed_folder)

    return final_processed_df

def resume_bids_aio(chunks_path:str):
    chunks = [os.path.join(chunks_path, f) for f in os.listdir(chunks_path) if f.endswith('.csv')]

    for i in chunks:
        apply_aio(
            df_path=i, 
            convert_to_nifti=False,
            clean=True,
            save=True,
            reg_template=mni, 
        )
        os.remove(i)

def bids_finalize(root_path:str):  
    processed_folder = os.path.join(root_path, 'processed')
    final_processed_df = concat_csv_files(processed_folder)

    return final_processed_df

