import argparse
import matplotlib.pyplot as plt
import re


def parse_args():
    parser = argparse.ArgumentParser(description='Parse and plot subrecord percentage val results from log file.')
    parser.add_argument('log_filepath', help='path to log file')
    return parser.parse_args()


def main(args):
    val_loss_kld_regex = re.compile(r'val_loss_kld\ *?(?P<val_loss_kld>[+-]?([0-9]*[.])?[0-9]+)')
    model_index_regex = re.compile(r'MODEL_(?P<model_index>\d)')
    x = list(map(float, '0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0'.split()))
    n_vals_per_model = len(x)

    with open(args.log_filepath, 'r') as f:
        content = f.read()
    model_indices = [int(g) for g in model_index_regex.findall(content)]
    metrics = [float(g[0]) for g in val_loss_kld_regex.findall(content)]
    # print(f'model_indices: {model_indices}')
    # print(f'metrics: {metrics}')
    for i, model_index in enumerate(model_indices):
        model_metrics = metrics[i * n_vals_per_model:(i + 1) * n_vals_per_model]
        plt.plot(x[:len(model_metrics)], model_metrics, label=f'fold {model_index}')
        print(f'{model_index}: {model_metrics}')
    plt.xlabel('Subrecord Percentage')
    plt.ylabel('Validation Loss KLD')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
