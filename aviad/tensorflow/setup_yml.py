import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/aviad.yaml', help="Which configuration to use. See into 'config' folder")

opt = parser.parse_args()
print(opt.config)

with open(opt.config, 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

print (config)