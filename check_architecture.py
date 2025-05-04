import argparse as ap
import logging

from utils import (model_init, load_model_params)


def get_arguments():

    parser = ap.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default=None, required=True)
    parser.add_argument('-r', '--rank', type=str, default=1)

    return parser.parse_args()


def print_model_layers(_model, indent: int = 0):

    for name, layer in _model.named_children():
        print(' ' * indent + f'{name}: {layer.__class__.__name__}')
        print_model_layers(layer, indent + 2)


if __name__ == "__main__":
    print()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    args = get_arguments()

    model_dir = f"./artifacts/classifier/{args.model}"
    load_params, checkpoint_classifier = load_model_params(model_dir, args.rank)
    load_params["masking"] = False
    logging.info(f"@params:\n\n{load_params}\n")

    # load model parameters
    model = model_init(**load_params)

    print_model_layers(model)
