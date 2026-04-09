import os
import pytest
from unittest.mock import MagicMock
import pandas as pd
from engine.tearsheet import TearsheetGenerator

def test_tearsheet_creation_and_saving(tmp_path):
    # Use pytest's tmp_path to isolate file creation
    base_dir = str(tmp_path)
    cartridge_name = "Test_Cartridge"
    
    generator = TearsheetGenerator(cartridge_name, base_dir=base_dir)
    
    # Verify directory was created dynamically
    expected_dir = os.path.join(base_dir, 'reports', cartridge_name, '15m')
    assert os.path.exists(expected_dir)
    assert os.path.isdir(expected_dir)
    
    # 1. Mock an Optuna Study
    mock_study = MagicMock()
    # Mock the trials_dataframe to return a DataFrame with some 'params_x', 'params_y', and 'value'
    mock_df = pd.DataFrame({
        'params_length': [10, 20, 30],
        'params_threshold': [0.1, 0.2, 0.3],
        'value': [1.5, 2.0, 1.2],
        'state': ['COMPLETE', 'COMPLETE', 'COMPLETE']
    })
    mock_study.trials_dataframe.return_value = mock_df
    
    # Generate Chart A (Optimization Surface - 2 Parameters)
    opt_path = generator.generate_optimization_chart(mock_study, metric_name="Profit Factor")
    assert opt_path is not None
    assert os.path.exists(opt_path)
    assert opt_path.endswith("optimization_surface.png")
    
    # 2. Mock Permutation Data
    synthetic_metrics = [1.0, 1.1, 0.9, 1.2, 0.8, 1.05]
    real_benchmark = 1.5
    p_value = 0.01
    
    # Generate Chart B (Permutation Distribution)
    perm_path = generator.generate_permutation_chart(synthetic_metrics, real_benchmark, p_value, metric_name="Profit Factor")
    assert perm_path is not None
    assert os.path.exists(perm_path)
    assert perm_path.endswith("permutation_distribution.png")

def test_optimization_chart_one_param(tmp_path):
    generator = TearsheetGenerator("OneParam_Cartridge", base_dir=str(tmp_path))
    
    mock_study = MagicMock()
    mock_df = pd.DataFrame({
        'params_length': [10, 20, 30],
        'value': [1.5, 2.0, 1.2],
        'state': ['COMPLETE', 'COMPLETE', 'COMPLETE']
    })
    mock_study.trials_dataframe.return_value = mock_df
    
    opt_path = generator.generate_optimization_chart(mock_study)
    assert os.path.exists(opt_path)
