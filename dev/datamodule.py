from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
import os
from torch.utils.data import DataLoader, TensorDataset
import torch

from src.e3foam.tensors.base import TensorData
from src.e3foam.preprocessing.scalers import EquivariantScalerWrapper, MeanScaler
from src.e3foam.foam.foamfield import parse_foam_file
from sklearn.preprocessing import StandardScaler

import dotenv

dotenv.load_dotenv()


class CFDCaseDataset(LightningDataModule):
    def __init__(self,
                 base_path,
                 train_cases,
                 val_cases,
                 test_cases,
                 fields,
                 labels,
                 batch_size=32,
                 use_scaler_x=True,
                 use_scaler_y=True):
        super().__init__()
        self.base_path = os.path.join(os.environ['PROJECT_ROOT'], base_path)
        self.train_cases = train_cases
        self.val_cases = val_cases
        self.test_cases = test_cases
        self.fields = fields
        self.labels = labels
        self.batch_size = batch_size

        # Initialize scalers with equivariant wrapper
        self.use_scaler_x = use_scaler_x
        self.use_scaler_y = use_scaler_y
        if self.use_scaler_x:
            self.scaler_x = EquivariantScalerWrapper(MeanScaler())
        else:
            self.scaler_x = None
        if self.use_scaler_y:
            self.scaler_y = EquivariantScalerWrapper(MeanScaler())
        else:
            self.scaler_y = None

        # Add irreps attributes
        self.x_irreps = None
        self.y_irreps = None

    @staticmethod
    def concat_fields(case_path, fields):
        """Concatenate multiple fields from a case into single TensorData."""
        case_tensordatas = []
        for field in fields:
            field_path = os.path.join(case_path, field)
            # First parse the foam file
            foam_field = parse_foam_file(field_path)
            
            # Determine symmetry from field type
            is_symmetric = 'symm' in foam_field.field_type.lower()
            is_skew = 'skew' in foam_field.field_type.lower()
            
            # Create TensorData with proper symmetry
            tensordata = TensorData(
                values=foam_field.internal_field,
                symmetry=1 if is_symmetric else (-1 if is_skew else 0)
            )
            case_tensordatas.append(tensordata)
        return TensorData.concat(case_tensordatas)

    @classmethod
    def append_cases(cls, base_path, case_names, fields):
        """Append data from multiple cases."""
        cases = []
        for case_name in case_names:
            case_path = os.path.join(base_path, case_name)
            case_data = cls.concat_fields(case_path, fields)
            cases.append(case_data)
        return TensorData.append(cases)

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare data for each stage (fit/test)."""
        # Load and concatenate training data
        self.train_tensordata_x = self.append_cases(self.base_path, self.train_cases, self.fields)
        self.train_tensordata_y = self.append_cases(self.base_path, self.train_cases, self.labels)

        # Load validation and test data
        self.val_tensordata_x = self.append_cases(self.base_path, self.val_cases, self.fields)
        self.val_tensordata_y = self.append_cases(self.base_path, self.val_cases, self.labels)
        self.test_tensordata_x = self.append_cases(self.base_path, self.test_cases, self.fields)
        self.test_tensordata_y = self.append_cases(self.base_path, self.test_cases, self.labels)

        # Only fit scalers here, don't transform
        if self.use_scaler_x:
            self.scaler_x.fit(self.train_tensordata_x)
        if self.use_scaler_y:
            self.scaler_y.fit(self.train_tensordata_y)

    def general_dataloader(self, step='train'):
        """Create dataloader for a specific step."""
        shuffle = (step == 'train')

        # Get appropriate tensors and transform them here
        x = getattr(self, f'{step}_tensordata_x')
        y = getattr(self, f'{step}_tensordata_y')

        if self.use_scaler_x:
            x = self.scaler_x.transform(x)
        if self.use_scaler_y:
            y = self.scaler_y.transform(y)

        # Convert to tensors and irreps
        x_tensor = x.to_tensor()
        y_tensor = y.to_tensor()

        # Store irreps on first call and get irrep tensors
        if self.x_irreps is None:
            x_irrep_tensor, self.x_irreps = x.to_irreps()  # Returns (tensor, irrep_str)
        else:
            x_irrep_tensor, _ = x.to_irreps()

        if self.y_irreps is None:
            y_irrep_tensor, self.y_irreps = y.to_irreps()
        else:
            y_irrep_tensor, _ = y.to_irreps()

        dataset = TensorDataset(x_irrep_tensor, y_irrep_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4)

    def train_dataloader(self):
        return self.general_dataloader(step='train')

    def val_dataloader(self):
        return self.general_dataloader(step='val')

    def test_dataloader(self):
        return self.general_dataloader(step='test')


if __name__ == "__main__":
    datamodule = CFDCaseDataset(
        'data/periodic-hills',
        ['0p5/0', '0p8/0', '1p0/0'],
        ['1p2/0'],
        ['1p5/0'],
        ['S', 'p', 'k'],
        ['Rdns']
    )
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    print(datamodule.x_irreps, datamodule.y_irreps)