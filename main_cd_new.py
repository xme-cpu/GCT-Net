import torch
import os
from argparse import Namespace  # Import Namespace to create a mock args object
from models.trainer import *
import utils  # Assuming utils is a local module

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

"""
the main function for training the CD networks
"""

def train(args):
    """Handles the training process."""
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()

def test(args):
    """Handles the evaluation process."""
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)
    model.eval_models()


if __name__ == '__main__':
    # ------------
    # Hardcoded Arguments
    # ------------
    # Create a Namespace object to hold all parameters
    args = Namespace()

    # Shell script variables
    args.gpu_ids = '0'
    args.checkpoint_root = 'checkpoints'
    args.data_name = 'WHU-CD'
    args.img_size = 256
    args.batch_size = 8

    # --- [!!] IMPORTANT CHANGES [!!] ---
    args.lr = 0.0001  # AdamW requires a much lower learning rate
    args.net_G = 'hybrid_bit'  # This is our new model's name
    args.optimizer = 'adamw'  # We must use AdamW
    # --- End of Changes ---

    args.max_epochs = 200
    args.lr_policy = 'linear'
    args.split = 'train'
    args.split_val = 'val'

    # Construct the project name as in the script
    args.project_name = (
        f"CD_{args.net_G}_{args.data_name}_b{args.batch_size}_"
        f"lr{args.lr}_{args.split}_{args.split_val}_"
        f"{args.max_epochs}_{args.lr_policy}"
    )

    # Default parameters from the original parser
    args.num_workers = 6
    args.dataset = 'CDDataset'
    args.n_class = 2
    args.loss = 'ce'
    # args.optimizer = 'sgd'
    args.lr_decay_iters = 100

    # --- End of Hardcoded Arguments ---

    # Set up device
    utils.get_device(args)
    print(f"Using GPU IDs: {args.gpu_ids}")
    print(f"Project Name: {args.project_name}")

    # Create checkpoints directory
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")

    # Create visualization directory
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {args.vis_dir}")

    # Start the training and testing process
    print("\n--- Starting Training ---")
    train(args)
    print("\n--- Training Finished ---")

    print("\n--- Starting Testing ---")
    test(args)
    print("\n--- Testing Finished ---")