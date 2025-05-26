import os
import argparse
import numpy as np
import torch
from PIL import Image
from model import PromptIR


def compute_pred_npz(input_folder, model, device, fp16=False):
    """
    Run inference with TTA and save results in a dict.

    Args:
        input_folder (str): path to degraded images
        model (nn.Module): PromptIR model, set to eval mode
        device (torch.device): computation device
        fp16 (bool): enable half-precision inference
    Returns:
        dict: mapping filename to restored image array (3, H, W) uint8
    """
    model.eval()
    images_dict = {}

    # Define 8 TTA transforms and their inverses
    tta_transforms = [
        (lambda x: x, lambda x: x),
        (lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
        (lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
        (lambda x: torch.rot90(x, 1, dims=[2, 3]),
         lambda x: torch.rot90(x, 3, dims=[2, 3])),
        (lambda x: torch.rot90(x, 2, dims=[2, 3]),
         lambda x: torch.rot90(x, 2, dims=[2, 3])),
        (lambda x: torch.rot90(x, 3, dims=[2, 3]),
         lambda x: torch.rot90(x, 1, dims=[2, 3])),
        (lambda x: torch.rot90(torch.flip(x, dims=[3]), 1, dims=[2, 3]),
         lambda x: torch.flip(torch.rot90(x, 3, dims=[2, 3]), dims=[3])),
        (lambda x: torch.rot90(torch.flip(x, dims=[3]), 3, dims=[2, 3]),
         lambda x: torch.flip(torch.rot90(x, 1, dims=[2, 3]), dims=[3])),
    ]

    for filename in sorted(os.listdir(input_folder)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load image, normalize, convert to tensor (1,3,H,W)
        path = os.path.join(input_folder, filename)
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))[None].to(device)
        if fp16:
            tensor = tensor.half()

        preds = []
        with torch.no_grad():
            for tf, inv in tta_transforms:
                out = model(tf(tensor))
                out = inv(out)
                preds.append(out)

        # Average predictions across TTA variants
        out_avg = torch.stack(preds, dim=0).mean(dim=0)
        out_np = out_avg.clamp(0, 1).cpu().float().numpy()[0]

        # Save as uint8 array, shape=(3, H, W)
        images_dict[filename] = (out_np * 255.0).round().astype(np.uint8)

    return images_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_folder',
        required=True,
        help='path to degraded test images'
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='path to trained .pth model checkpoint'
    )
    parser.add_argument(
        '--output_npz',
        default='pred.npz',
        help='filename for output .npz'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='use half-precision for inference'
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = PromptIR(decoder=True).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    if args.fp16 and device.type == 'cuda':
        model.half()

    # Run inference with TTA
    outputs = compute_pred_npz(
        args.input_folder, model, device, fp16=args.fp16
    )
    np.savez(args.output_npz, **outputs)
    print(f"Saved {len(outputs)} images to {args.output_npz}")


if __name__ == '__main__':
    main()
