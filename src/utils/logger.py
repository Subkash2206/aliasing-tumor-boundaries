import os

# Initialize wandb placeholder, users will provide their own keys
def init_wandb_logger(project_name="spectral-aliasing-brats", entity=None, config=None):
    """
    Initializes a Weights & Biases logger.
    """
    try:
        import wandb
    except ImportError:
        print("wandb not installed. Run `pip install wandb`.")
        return None

    if os.environ.get("WANDB_API_KEY") is None:
        print("Warning: WANDB_API_KEY is not set. wandb will run in offline mode.")
        mode = "offline"
    else:
        mode = "online"
        
    wandb.init(project=project_name, entity=entity, config=config, mode=mode)
    return wandb
