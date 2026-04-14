import os
import argparse
import torch
from monai.networks.nets import SegResNet
from monai.losses import DiceLoss
from src.data.transforms_3d import get_brats_3d_train_transforms, get_brats_3d_val_transforms
from src.data.brats_3d_dataset import get_3d_dataloaders
from src.models.blurpool3d import replace_stride_with_blurpool3d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1) # 1 for 6GB VRAM
    parser.add_argument("--accumulate_grad_batches", type=int, default=4) # Virtual batch size 4
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--use_blurpool", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'BraTS2021_Training_Data')

    try:
        print("Loading dataset...")
        train_transforms = get_brats_3d_train_transforms(patch_size=(64, 64, 64))
        val_transforms = get_brats_3d_val_transforms()
        train_loader, val_loader, _, _ = get_3d_dataloaders(
            data_dir=data_dir,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            batch_size=args.batch_size,
            num_workers=0
        )
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # SOTA Architecture
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=4,  # background, ncr/wt, edema, et
        dropout_prob=0.2,
    ).to(device)

    if args.use_blurpool:
        print("Injecting 3D BlurPool into SegResNet...")
        replace_stride_with_blurpool3d(model)
        model = model.to(device)

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    best_metric = -1
    best_metric_epoch = -1

    scaler = torch.amp.GradScaler('cuda')

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        step = 0
        optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{args.epochs}")
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["seg"].to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                # Normalize loss by accumulation steps
                loss = loss / args.accumulate_grad_batches

            scaler.scale(loss).backward()
            
            # Gradient Accumulation
            if step % args.accumulate_grad_batches == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            epoch_loss += loss.item() * args.accumulate_grad_batches
            print(f"  Step {step}/{len(train_loader)} - Batch Loss: {loss.item() * args.accumulate_grad_batches:.4f}")
            
        print(f"Train Loss: {epoch_loss/step:.4f}")

        # Simple validation: just measure loss or save latest for now,
        # Real validation will use evaluate_spectral_decay.py
        if (epoch + 1) % args.val_interval == 0:
            torch.save(model.state_dict(), os.path.join("results", f"latest_segresnet_bp_{args.use_blurpool}.pth"))
            print(f"Saved latest checkpoint for epoch {epoch + 1}.")

if __name__ == "__main__":
    main()
