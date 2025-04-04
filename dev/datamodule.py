from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
import os
from torch.utils.data import DataLoader, TensorDataset
import torch

import dotenv
import sys

# Get the directory containing this script
this_dir = os.path.dirname(os.path.abspath(__file__))

# Go one directory up — that's your submodule root
submodule_root = os.path.abspath(os.path.join(this_dir, 'e3foam'))

# Add it to sys.path if not already there
if submodule_root not in sys.path:
    sys.path.insert(0, submodule_root)

from src.e3foam.tensors.base import TensorData
from src.e3foam.preprocessing.scalers import EquivariantScalerWrapper, MeanScaler
from src.e3foam.foam.foamfield import parse_foam_file
from sklearn.preprocessing import StandardScaler

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
        self.x_cartesian = None
        self.y_cartesian = None
        
        # Add rizzler attributes
        self.x_irrep_rizzler = None
        self.y_irrep_rizzler = None
        self.x_cartesian_rizzler = None
        self.y_cartesian_rizzler = None

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
                tensor=foam_field.internal_field,
                symmetry=1 if is_symmetric else (-1 if is_skew else 0),
                is_flattened=True
            )
            case_tensordatas.append(tensordata)
        return TensorData.cat(case_tensordatas)

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
            
        # Create rizzlers for x and y data
        # We use the training data to create the rizzlers, but they'll work for all datasets
        # with the same structure
        self.x_irrep_rizzler = self.train_tensordata_x.get_irrep_rizzler()
        self.y_irrep_rizzler = self.train_tensordata_y.get_irrep_rizzler()
        
        # Get irreps information for reference
        _, self.x_irreps, self.x_cartesian = self.train_tensordata_x.to_irreps()
        _, self.y_irreps, self.y_cartesian = self.train_tensordata_y.to_irreps()
        
        # Create cartesian rizzlers with caching enabled for better performance
        self.x_cartesian_rizzler = self.train_tensordata_x.get_cartesian_rizzler(cache_rtps=True)
        self.y_cartesian_rizzler = self.train_tensordata_y.get_cartesian_rizzler(cache_rtps=True)

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
            
        # Convert to irreps using rizzlers
        x_irrep_tensor = self.x_irrep_rizzler(x.tensor)[0]
        y_irrep_tensor = self.y_irrep_rizzler(y.tensor)[0]
        
        dataset = TensorDataset(x_irrep_tensor, y_irrep_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4)

    def train_dataloader(self):
        return self.general_dataloader(step='train')
    
    def val_dataloader(self):
        return self.general_dataloader(step='val')

    def test_dataloader(self):
        return self.general_dataloader(step='test')

    def to_cartesian(self, x_irrep_tensor, y_irrep_tensor=None):
        """Convert irrep tensors back to Cartesian representation.
        
        Args:
            x_irrep_tensor: Input features in irrep representation
            y_irrep_tensor: Optional target values in irrep representation
            
        Returns:
            Tensor(s) in Cartesian representation
        """
        x_cartesian = self.x_cartesian_rizzler(x_irrep_tensor)
        
        if y_irrep_tensor is not None:
            y_cartesian = self.y_cartesian_rizzler(y_irrep_tensor)
            return x_cartesian, y_cartesian
        
        return x_cartesian


if __name__ == "__main__":
    datamodule = CFDCaseDataset(
        'periodic-hills',
        ['0p5\\0', '0p8\\0', '1p0\\0'],
        ['1p2\\0'],
        ['1p5\\0'],
        ['S', 'p', 'k'],
        ['Rdns'],
        use_scaler_x=True,
        use_scaler_y=True
    )
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    print(datamodule.x_irreps, datamodule.y_irreps)
    
    # Test the rizzlers
    for x_batch, y_batch in train_dataloader:
        # Convert back to Cartesian
        x_cart, y_cart = datamodule.to_cartesian(x_batch, y_batch)
        print(f"Irrep shape: {x_batch.shape}, Cartesian shape: {x_cart.shape}")
        break







