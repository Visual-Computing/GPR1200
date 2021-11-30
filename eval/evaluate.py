import argparse
import numpy as np
import pickle
from GPR1200 import GPR1200


def add_parser_arguments(parser):
    
    parser.add_argument(
        "--evalfile-path",
        metavar="EVALFILE",
        help="Path to embeddings or indices file"
    )

    parser.add_argument(
        "--dataset-path",
        metavar="DSFILEPATH",
        help="Path to the GPR1200 images folder"
    )

    parser.add_argument(
        "--mode",
        metavar="MODE",
        default="embeddings",
        choices=["embeddings", "indices"],
        help="Run this script in embeddings mode if you have one embedding per image \
        or indices mode with precomputed nearest neighbour indices in any other case",
    )

def main(args):

    GPR1200_dataset = GPR1200(args.dataset_path)

    try:
        print("Trying to load evalfile with numpy")
        evalfile = np.load(args.evalfile_path)
        print("Load succesfull. Data shape:", evalfile.shape)
    except:
        print("Numpy load failed, falling back to pickle")

        try: 
            with open(args.evalfile_path,'rb') as f:
                evalfile = pickle.load(f)
                evalfile = np.array(evalfile)
                print("Load succesfull. Data shape:", evalfile.shape)
        except:
            raise ValueError("Invalid evalfile. \
            Make sure that the file can be loaded with either numpy or pickle")

    if args.mode == "embeddings":
        results = GPR1200_dataset.evaluate(features=evalfile, compute_partial=True)
    elif args.mode == "indices":
        results = GPR1200_dataset.evaluate(indices=evalfile, compute_partial=True)
    else:
        raise ValueError("Invalid mode selected. \
        Options are embeddings and indices")
    
    print()
    print("---------Results:")
    gpr, lm, iNat, ims, instre, sop, faces = results
    print("GPR1200 mAP: {}".format(gpr))
    print("Landmarks: {}, IMSketch: {}, iNat: {}, Instre: {}, SOP: {}, faces: {}".format(lm, ims, iNat, instre, sop, faces))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="GPR1200 Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    add_parser_arguments(parser)

    args, rest = parser.parse_known_args()

    main(args)
