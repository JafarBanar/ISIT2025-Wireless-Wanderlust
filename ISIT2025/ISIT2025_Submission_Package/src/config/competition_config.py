"""Competition-specific configuration settings."""

# Competition Dataset Configuration
DATASET_CONFIG = {
    'n_arrays': 4,  # 4 remote antenna arrays
    'n_elements': 8,  # 8 elements per array
    'n_frequencies': 16,  # 16 frequency bands
    'n_samples': 10000  # Number of samples for validation
}

# Model Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 1e-3,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'min_lr': 1e-6
}

# Vanilla Model Configuration
VANILLA_MODEL_CONFIG = {
    'dropout_rate': 0.3,
    'l2_reg': 0.01
}

# Trajectory Model Configuration
TRAJECTORY_MODEL_CONFIG = {
    'sequence_length': 10,
    'lstm_units': 128,
    'attention_heads': 4,
    'dropout_rate': 0.3,
    'l2_reg': 0.01
}

# Feature Selection Model Configuration
FEATURE_SELECTION_CONFIG = {
    'sequence_length': 10,
    'n_features_to_select': 32,
    'n_priority_levels': 4,
    'dropout_rate': 0.3,
    'l2_reg': 0.01
}

# Competition Metrics Configuration
METRICS_CONFIG = {
    'mae_weight': 0.7,  # Weight for MAE in combined metric
    'r90_weight': 0.3,  # Weight for R90 in combined metric
}

# Paths Configuration
PATHS = {
    'data_dir': 'data/competition',
    'output_dir': 'outputs/validation',
    'documentation_dir': 'docs/competition'
}

# Competition Deadlines
DEADLINES = {
    'registration': '2025-05-07',
    'submission': '2025-06-13'
} 