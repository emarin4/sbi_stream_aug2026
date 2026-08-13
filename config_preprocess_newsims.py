"""Configuration file for training NPE on particle datasets."""
import os
import numpy as np
from ml_collections import ConfigDict


def get_config():
    """Get the default configuration for training NPE on particle data."""

    config = ConfigDict()
    config.root = '/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/sims'
    config.name = 'sims_pert_813_mocks'
    config.root_out = '/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/preprocessed'
    config.name_out = 'preprocessed_813_mocks'
    config.labels = ['log_mass', 'log_scale_radius', 'phi1_impact_today', 'time_impact','log_impact_parameter', 'v_rel_para', 'v_rel_perp', 'angle_pos_at_impact', 'delta_angle','angle_vel_at_impact', 'delta_phi1']

    config.features = ['phi1', 'phi2', 'vr', 'pm1', 'pm2', 'dist']

    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    config.start_dataset = task_id
    
    #task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    #config.start_dataset = task_id 
    config.num_datasets = 1 

    config.num_subsamples = 1 #each stream gets sampled once # at least 5 but ideally > 10
    #config.num_per_subsample = #int(np.random.randint( 100, 201, size=config.num_subsamples))
    config.num_per_subsample_min = None
    config.num_per_subsample_max = None
    config.phi1_min = -20
    config.phi1_max = 16 #16
    config.uncertainty_model = None#'aau'
    config.include_uncertainty = False #True
    return config
