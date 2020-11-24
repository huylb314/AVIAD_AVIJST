import yaml

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', default='/w/31/faghri/vsepp_data/',
    #                     help='path to datasets')
    # parser.add_argument('--data_name', default='precomp',
    #                     help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    # parser.add_argument('--crop_size', default=224, type=int,
    #                     help='Size of an image crop as the CNN input.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--test_step', default=100000000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads model, optimizer, scheduler')
    parser.add_argument('--load-model', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads only the model')
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--reinitialize-scheduler', action='store_true', help='Reinitialize scheduler. To use with --resume')
    parser.add_argument('--config', type=str, help="Which configuration to use. See into 'config' folder")

    opt = parser.parse_args()
    print(opt)

with open(opt.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)