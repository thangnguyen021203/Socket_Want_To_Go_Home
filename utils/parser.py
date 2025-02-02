import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # # Model parameters
    # parser.add_argument('--model', type=str, default='resnet',
    #                     help='model name (default: resnet)')
    # parser.add_argument('--input_size', type=int, default=224,
    #                     help='input size for model (default: 224)')
    # parser.add_argument('--num_classes', type=int, default=1000,
    #                     help='number of classes (default: 1000)')
    # parser.add_argument('--pretrained', type=bool, default=True,
    #                     help='pretrained model on ImageNet (default: True)')
    # parser.add_argument('--checkpoint', type=str, default=None,
    #                     help='path to load checkpoint (default: None)')

    # Data parameters
    parser.add_argument('--client_id', type=int, default=0,
                        help='Client ID (default: client 0)')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Dataset name (default: MNIST)')
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to non-IID. Set to 1 for IID.')
    # parser.add_argument('--data_path', type=str, default='data',
    #                     help='path to data (default: data)')
    # parser.add_argument('--batch_size', type=int, default=128,
    #                     help='input batch size for training (default: 128)')
    # parser.add_argument('--num_workers', type=int, default=4,
    #                     help='number of workers (default: 4)')

    # # Device parameters
    # parser.add_argument('--device', type=str, default='cuda',
    #                     help='device (default: cuda)')

    return parser.parse_args()