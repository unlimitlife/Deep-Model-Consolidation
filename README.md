# Incremental-Learning

In refactoring the incremental learning methods for collaborative implementation.

Implementation includes:
1. Class-incremental Learning via Deep Model Consolidation (WACV2020)
## Code

### Requirements
- python 3.6+
- cuda 9.0+

### Datasets
- cifar100

### Usage
1. git clone or download this repository
2. change ```config.py``` **save_path** and **data_path**
3. ```source {your_venv_directory}```
5. ```python3 main.py "model" "dataset" "curriculum" "method" --option```

- positional arguments:
  - model : {resnet18, resnet20, resnet32, resnet56}
  - dataset : {cifar100}
  - curriculum : {basic, rand1, rand2, base1, base2}
  - method : {DMC}
  
- optional arguments:
  - -h, --help
  - --desc
  - --classes-per-task
  - --memory-cap

**if you need more information, run ```python3 main.py -h```**

### Modify
To add a new method:
#### Simple
1. modify ```method/our_method``` package.
2. It's okay to modify anything, but ```method/our_method/train.py/train``` should remain.

#### Not Simple
1. make new package in ```method```
2. ```__init__.py``` file must import train function
3. your method must include a ```def train(config, method_config, data_config)```
4. add your method to ```config.py``` file
5. add your method to ```parser.py``` file

The structure of **config**, **data_config**, **method_config** is as follow.\
if you need more variable, modify ```main.py``` and ```config.py```

    config (dict): config file dictionary.
        model (str): name of network. [selector.model(model, ...)]
        classes_per_task (int): classes per task.
        DA (bool): if True, apply data augment.
        memory_cap (int): sample memory size.
        num_workers (int): how many subprocesses to use for data loading.
                           0 means that the data will be loaded in the main process. (default: 0)
        batch_size (int): how many samples per batch to load. (default: 1)
        device (torch.device): gpu or cpu.
        data_path (str): root directory of dataset.
        save_path (str): directory for save. (not taskwise)
    data_config (dict): data config file dictionary.
        dataset (str): name of dataset.
        total_classes (int): total class number of dataset.
        transform (dict):
            train (transforms): transforms for train dataset.
            test (transforms): transforms for test dataset.
        curriculums (list): curriculum list.
        classes (list): class name list.
    method_config (dict): method config file dictionary.
        method (str): name of method.
        process_list (list): process list.
        package (string): current package name.

### Save File
##### File Name
1. {number}.pth.tar: end of the process.
4. LastModel.pth.tar: end of the task.

##### In Save File
- epoch: epoch
- state_dict: network state dict
- single_total_accuracy: Total accuracy based on single-head.
- single_class_accuracy: Class-wise accuracy based on single-head.
- multi_total_accuracy: Total accuracy based on multi-head.
- multi_class_accuracy: Class-wise accuracy based on multi-head.

