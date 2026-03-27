import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def load_stratified_split(data_dir, folders, test_size=0.2, random_state=42):
    """
    Looks for a metadata CSV (e.g., BraTS2021_metadata.csv or train_labels.csv) 
    to stratify by grade. Falls back to deterministic split with warning if missing.
    """
    csv_paths = [
        os.path.join(data_dir, "BraTS2021_metadata.csv"),
        os.path.join(data_dir, "train_labels.csv"),
        os.path.join(os.path.dirname(data_dir), "BraTS2021_metadata.csv"),
        os.path.join(os.path.dirname(data_dir), "train_labels.csv")
    ]
    
    metadata = None
    csv_found = False
    for cp in csv_paths:
        if os.path.exists(cp):
            try:
                metadata = pd.read_csv(cp)
                print(f"Loaded stratification metadata from {cp}")
                csv_found = True
                break
            except Exception as e:
                print(f"Error reading {cp}: {e}")
                
    grades = []
    
    # Try to map case IDs to grades
    if metadata is not None and ('Grade' in metadata.columns or 'grade' in metadata.columns):
        grade_col = 'Grade' if 'Grade' in metadata.columns else 'grade'
        id_col = None
        for col in ['BraTS2021', 'Subject', 'ID', 'id']:
            if col in metadata.columns:
                id_col = col
                break
                
        if id_col is not None:
            id_to_grade = dict(zip(metadata[id_col].astype(str), metadata[grade_col]))
        else:
            id_to_grade = {}
            
        for folder in folders:
            case_id = os.path.basename(folder)
            if case_id in id_to_grade:
                grades.append(id_to_grade[case_id])
            elif case_id.split("_")[-1] in id_to_grade:
                grades.append(id_to_grade[case_id.split("_")[-1]])
            else:
                grades.append("Unknown")
    else:
        if not csv_found:
            print("WARNING: No stratification CSV found. Falling back to deterministic random split (seed 42).")
        for folder in folders:
            grades.append("Unknown")
            
    if all(g == "Unknown" for g in grades):
        return train_test_split(folders, test_size=test_size, random_state=random_state)
    else:
        return train_test_split(folders, test_size=test_size, random_state=random_state, stratify=grades)

def get_brats_manifest(data_dir, test_size=0.2, random_state=42):
    """
    Scans the BraTS 2021 directory and returns a list of dictionaries 
    containing paths for all 4 modalities and the segmentation mask.
    """
    folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, f)) and f.startswith("BraTS2021_")]
               
    if len(folders) == 0:
        return [], []
        
    train_folders, val_folders = load_stratified_split(data_dir, folders, test_size, random_state)
    
    def process_folders(folder_list):
        manifest = []
        for folder in sorted(folder_list):
            case_id = os.path.basename(folder)
            t1 = os.path.join(folder, f"{case_id}_t1.nii.gz")
            t1ce = os.path.join(folder, f"{case_id}_t1ce.nii.gz")
            t2 = os.path.join(folder, f"{case_id}_t2.nii.gz")
            flair = os.path.join(folder, f"{case_id}_flair.nii.gz")
            seg = os.path.join(folder, f"{case_id}_seg.nii.gz")
            
            if all(os.path.exists(p) for p in [t1, t1ce, t2, flair, seg]):
                manifest.append({
                    'id': case_id,
                    't1': t1,
                    't1ce': t1ce,
                    't2': t2,
                    'flair': flair,
                    'seg': seg
                })
        return manifest
        
    return process_folders(train_folders), process_folders(val_folders)


class BraTS2DSliceDataset(Dataset):
    """
    A PyTorch Dataset that takes a list of 3D volume paths and 
    returns a specific 2D axial slice (e.g. 240x240) and its mask.
    """
    def __init__(self, manifest, transform=None, num_slices_per_volume=155):
        self.manifest = manifest
        self.transform = transform
        self.num_slices_per_volume = num_slices_per_volume
        
    def __len__(self):
        return len(self.manifest) * self.num_slices_per_volume
        
    def __getitem__(self, idx):
        vol_idx = idx // self.num_slices_per_volume
        slice_idx = idx % self.num_slices_per_volume
        
        data_dict = self.manifest[vol_idx].copy()
        data_dict['slice_idx'] = slice_idx
        
        if self.transform:
            data_dict = self.transform(data_dict)
            
        return data_dict['image'], data_dict['seg']
