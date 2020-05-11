import os
import sys
import shutil
import importlib
import copy
from utils import save_pyplot, save_reliability_diagram
from dataset import taskset
from Logger import Logger
from parser import args, command
from config import config, method_config, data_config



# select model / data / curriculum / method
model = args.model
data_config = data_config[args.dataset]
data_config['curriculums'] = data_config['curriculums'][args.curriculum]
method_config = method_config[args.method]

# preprocess dataset, If the dataset does not exist, download it and create a dataset.
taskset.preprocess(config['data_path'], data_config['dataset'])

'''
Assign path
data_path = config['data_path']/il+{dataset}
save_path = config['save_path']/{model}/[da_]+{dataset}/{curriculum}/{memory_cap}/{method}/{desc} + /task--
'''
config['data_path'] = os.path.join(config['data_path'], 'il' + data_config['dataset'])
config['save_path'] = os.path.join(config['save_path'],
                                   args.model,
                                   args.dataset,
                                   args.curriculum,
                                   str(args.memory_cap),
                                   args.method,
                                   args.desc)

if os.path.exists(config['save_path']):
    print("Directory is already exists!", file=sys.stderr)
    sys.exit()

'''
add data to config dictionary
if you want to add more parameters, add them here!
'''
######################################################
config['model'] = model
config['classes_per_task'] = args.classes_per_task
config['memory_cap'] = args.memory_cap
config['padding'] = args.padding
config['stride'] = args.stride
logger = Logger(os.path.join(config['save_path'], 'log'))

######################################################
shutil.copyfile(os.path.join(os.getcwd(), 'config.py'), os.path.join(config['save_path'], 'config.py'))
with open(os.path.join(config['save_path'], 'args'), 'w') as f:
    f.write(command)

package = importlib.import_module(method_config['package'])

package.train(copy.deepcopy(config), copy.deepcopy(method_config), copy.deepcopy(data_config), logger)

save_pyplot(config['save_path'], args.classes_per_task, data_config['total_classes'])
