import argparse
import denoisers
import tifffile
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False) 

    # Add data arguments
    parser.add_argument(
        "--model",
        default="UDVD",
        help="one of UDVD, N2N, N2S or UDVD_sf")
    parser.add_argument(
        "--data",
        default="data",
        help="path to .tif file to be denoised")
    parser.add_argument(
        "--num-epochs",
        default=50,
        type=int,
        help="epochs for the training. Default is 500")
    parser.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="train batch size")
    parser.add_argument(
        "--image-size",
        default=256,
        type=int,
        help="size of the patch")
    parser.add_argument(
        "--save-format",
        default="tif",
        help="save the denoised result as tif or npy. Default is tif")
    parser.add_argument(
        "--transforms",
        dest='feature',
        action='store_true')
    parser.add_argument(
        "--no-transforms",
        dest='feature', 
        action='store_false')
    parser.set_defaults(transforms=True)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()

    denoiser = denoisers.get_denoiser(args)
    
    model = denoiser.train(args)
    denoised = denoiser.denoise(model, args)
    
    if args.save_format == "tif":
        with tifffile.TiffWriter(f"{args.data.split('.')[0]}_{args.model.lower}.tif") as stack:
            stack.write(denoised)
    elif args.save_format == "npy":
        np.save(f"{args.data.split('.')[0]}_{args.model.lower}.npy", denoised)
