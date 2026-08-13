
from ml_collections import ConfigDict
import numpy as np


def get_config():
    config = ConfigDict()

    # seeding
    config.seed_data = 43
    config.seed_training = int(np.random.randint(1,100_000_000))

    # data configuration
    config.data = ConfigDict()
    config.data.data_type = 'preprocessed' #already subsampled 
    config.data.root = '/gpfs/projects/dirac/emarin4/sims'
    config.data.name = 'preprocessed_812_1_000_000'
    config.data.features = ['phi1', 'phi2', 'vr', 'pm1', 'pm2', 'dist']
    config.data.labels = ['log_mass', 'log_scale_radius'] # , 'phi1_impact_today', 'time_impact',
                      #'log_impact_parameter', 'v_rel_para', 'v_rel_perp', 
                      #'angle_pos_at_impact', 'delta_angle']
    config.data.num_datasets = 100 #38 
    config.data.start_dataset = 0
    config.data.num_subsamples = 1 #atleast 10  , 
    config.train_frac = 0.8
    config.num_workers = 0

    ## LOGGING AND WANDB CONFIGURATION ###
    config.workdir = '/gpfs/projects/dirac/emarin4/sbi_stream/graph_npe'
    config.wandb_project = 'sbi_aau_stepbystep' #change name
    config.entity = "desc_sbi_stream"
    config.tags = ['npe', 'cheb_conv']
    config.debug = False
    config.checkpoint = None  # Path to NPE checkpoint for resuming
    config.reset_optimizer = False
    config.enable_progress_bar = False
    config.log_model = 'all'  # Log model checkpoints to WandB

    # Reuse norm_dict from embedding checkpoint
    # if import embedding_nn from a pre-trained model, this should always be True
    config.reuse_embedding_norm_dict = False

    ### MODEL CONFIGURATION ###
    config.model = model = ConfigDict()
    model.input_size =  len(config.data.features) #9 #CHANGED for no uncert #len(config.data.features) * 2 #multiply by 2 for uncertainties
    model.output_size = len(config.data.labels)

    # Embedding network configuration
    model.embedding = ConfigDict()
    model.embedding.gnn = ConfigDict()
    model.embedding.gnn.projection_size = 128
    model.embedding.gnn.hidden_sizes = [128, ] * 4 #try 128, 64, and 3, 4, 5, layers, use 128 in both spots 
    model.embedding.gnn.graph_layer = 'ChebConv'
    model.embedding.gnn.graph_layer_params = ConfigDict()
    model.embedding.gnn.graph_layer_params.K = 8 #try 4, 6, and 8
    #model.embedding.gnn.graph_layer_params.sym = True
    model.embedding.gnn.pooling = 'mean'
    model.embedding.gnn.layer_norm = True
    model.embedding.gnn.norm_first = False
    model.embedding.gnn.act_name = 'leaky_relu'
    model.embedding.gnn.act_args = {'negative_slope': 0.01}

    # MLP configuration
    model.embedding.mlp = ConfigDict()
    model.embedding.mlp.hidden_sizes = [128, ] * 5 #try 128, 64, and 3, 4, 5 layers
    model.embedding.mlp.output_size = 12 #12
    model.embedding.mlp.dropout = 0.1
    model.embedding.mlp.batch_norm = True
    model.embedding.mlp.act_name = 'leaky_relu'
    model.embedding.mlp.act_args = {'negative_slope': 0.01}

    # NPE Normalizing Flows configuration
    model.flows = ConfigDict()
    model.flows.num_transforms = 6
    model.flows.hidden_features = [128, 128]
    model.flows.num_bins = 8
    model.flows.activation = 'tanh'
    model.flows.randperm = True

    # Pre-transformation configuration
    # Note: For NPE, pre_transforms are passed to NPE, not to embedding_nn
    config.pre_transforms = pre_transforms = ConfigDict()
    pre_transforms.apply_graph = True
    pre_transforms.graph_name = 'adaptive_knn'
    pre_transforms.graph_args = {'ratio':0.2, 'loop': True} #try 5, 10, 20 


    ### VISUALIZATION CALLBACK CONFIGURATION ###
    config.enable_visualization_callback = True
    config.visualization = visualization = ConfigDict()
    visualization.n_posterior_samples = 500
    visualization.n_val_samples = 200
    visualization.plot_every_n_epochs = 1
    visualization.plot_tarp = True
    visualization.plot_median_v_true = True
    visualization.plot_rank = True

### OPTIMIZER AND SCHEDULER CONFIGURATION ###
    config.optimizer = optimizer = ConfigDict()
    optimizer.name = "AdamW"
    optimizer.lr = 5e-4
    optimizer.betas = [0.9, 0.999]
    optimizer.weight_decay = 0.01

    config.scheduler = scheduler = ConfigDict()
    scheduler.name = "WarmUpCosineAnnealingLR"
    # Stepped per epoch, so these are epochs; decay_steps == num_epochs
    # so the cosine finishes exactly at the end of training. eta_min is
    # a multiplier on the base lr, not a floor: 2e-3 -> a final 1e-6.
    scheduler.decay_steps = 40
    scheduler.warmup_steps = 2
    scheduler.eta_min = 2e-3
    scheduler.interval = 'epoch'
    scheduler.restart = False
    scheduler.T_mult = 1

    ### TRAINING configuration ###
    config.accelerator = 'gpu'
    config.train_batch_size = 256
    config.eval_batch_size = 256
    config.num_epochs = 40
    config.num_steps = -1
    config.patience = 100
    config.gradient_clip_val = 0.5
    config.save_top_k = 5



    # ### OPTIMIZER AND SCHEDULER CONFIGURATION ###
    # config.optimizer = optimizer = ConfigDict()
    # optimizer.name = "AdamW"
    # optimizer.lr = 1e-4
    # optimizer.betas = [0.9, 0.999]
    # optimizer.weight_decay = 0.01

    # config.scheduler = scheduler = ConfigDict()
    # #scheduler.name = None
    # scheduler.name = "WarmUpCosineAnnealingLR"
    # scheduler.decay_steps = int((1_000_000) * 40 / 128) #int(100000 * 0.8 /128 * 20)
    # scheduler.warmup_steps = int(0.05 * scheduler.decay_steps)
    # scheduler.eta_min = 1e-6
    # scheduler.interval = 'step'
    # scheduler.restart = False
    # scheduler.T_mult = 1

    # ### TRAINING CONFIGURATION ###
    # config.accelerator = 'gpu'
    # config.train_batch_size = 128 #no of streams in grad calculation, check original 
    # config.eval_batch_size = 128
    # config.num_epochs = -1 #number of passes through all datasets (40, 50, ... 100) 
    # config.num_steps = scheduler.decay_steps #
    # config.patience = 25
    # config.gradient_clip_val = 1.0
    # config.save_top_k = 5

    return config
