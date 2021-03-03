import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default = 500, type = int)
    parser.add_argument('--learning-rate', default = 1e-3, type = float)
    parser.add_argument('--epochs', default = 10, type = int)
    parser.add_argument('--out-dim', default = 20, type = int)
    return parser.parse_args()