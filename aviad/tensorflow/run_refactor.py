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
    parser.add_argument('--config', type=str, default="configs/1k.yaml"\
                        help="Which configuration to use. See into 'config' folder")
                        
    opt = parser.parse_args()
    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)
        