import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from torchvision import transforms

class PatchDataset(Dataset):
    def __init__(self, csv_file, image_dir, subset_ids=None, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with patch names and labels.
            image_dir (str): Directory with all the patch images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.data_frame = pd.read_csv(csv_file)
        if subset_ids is not None:
            self.data_frame = self.data_frame[self.data_frame['TCGA_ID'].isin(subset_ids)]
            self.image_dir = image_dir
            self.transform = transform if transform else transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get patch filename and labels

        patch_name = self.data_frame.iloc[idx]['patch_filename']
        img_path = os.path.join(self.image_dir, patch_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Load labels
        tcga_id = self.data_frame.iloc[idx]['TCGA_ID']
        survival_time = torch.tensor(self.data_frame.iloc[idx]['Survival months'], dtype=torch.float32)
        event = torch.tensor(self.data_frame.iloc[idx]['censored'], dtype=torch.float32)

        # Robust way to handle Grade field safely
        grade_raw = self.data_frame.iloc[idx]['grade']
        try:
            grade_clean = int(float(grade_raw)) if pd.notna(grade_raw) else 0
        except:
            grade_clean = 0  # Default to 0 if completely invalid
        grade = torch.tensor(grade_clean, dtype=torch.long)

        return image, survival_time, event, grade, tcga_id
