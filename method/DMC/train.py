import selector
import os
import copy
import torch.utils.data
import utils
from dataset import taskset
from torchvision.datasets import ImageFolder
import torch.nn as nn
from .test import test, make_heatmap


def train(config, method_config, data_config, logger):
    """
    Args:
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
        logger (Logger): logger for the tensorboard.
    """
    model = config['model']
    classes_per_task = config['classes_per_task']
    memory_cap = config['memory_cap']
    device = config['device']
    data_path = config['data_path']
    external_data_path = method_config['external_data_path']
    external_cifar_data_path = method_config['external_cifar_data_path']
    save_path = config['save_path']

    total_classes = data_config['total_classes']
    train_transform = data_config['transform']['train']
    test_transform = data_config['transform']['test']
    curriculum = data_config['curriculums']
    dataset = data_config['dataset']

    

    '''Split curriculum [[task0], [task1], [task2], ...]'''
    curriculum = [curriculum[x:x + classes_per_task] for x in range(0, total_classes, classes_per_task)]

    '''Make sample memory'''
    sample_memory = taskset.SampleMemory(data_path, total_classes, len(curriculum), curriculum,
                                         transform=train_transform, capacity=memory_cap)

    external_taskset = []
    if dataset == 'cifar100':
        external_data_transform = method_config['external_data_transform']
        external_taskset = ImageFolder(os.path.join(external_data_path,'train'), transform=external_data_transform)
    if dataset == 'cifar10':
        external_data_transform = method_config['external_cifar_data_transform']
        external_taskset = ImageFolder(os.path.join(external_cifar_data_path,'train'), transform=external_data_transform)

    test_task = []
    train_task = []
    old_net = None
    cur_net = None
    '''Taskwise iteration'''
    for task_idx, task in enumerate(curriculum):
        train_task = task
        train_taskset = taskset.Taskset(data_path, train_task, task_idx, train=True, transform=train_transform)
        val_taskset = taskset.Taskset(data_path, train_task, task_idx, train=False, transform=test_transform)
        test_task.extend(task)
        test_taskset = taskset.Taskset(data_path, test_task, 0, train=False, transform=test_transform)

        '''Make directory of current task'''
        if not os.path.exists(os.path.join(save_path, 'task%02d' % task_idx)):
            os.makedirs(os.path.join(save_path, 'task%02d' % task_idx))

        config['task_path'] = os.path.join(save_path, 'task%02d' % task_idx)
        config['cml_classes'] = len(test_task)
        
        '''Make network'''
        net = selector.model(model, device, classes_per_task)
        if task_idx == 0:
            old_net = _train(task_idx, net, train_taskset, sample_memory, val_taskset, config, method_config, data_config, logger)
        else:
            cur_net = _train(task_idx, net, train_taskset, sample_memory, val_taskset, config, method_config, data_config, logger)
        
        if cur_net is not None:
            old_net = consolidation(task_idx, old_net, cur_net, external_taskset, test_taskset, config, method_config, data_config, logger)

        sample_memory.update()
        torch.save(sample_memory, os.path.join(config['task_path'], 'sample_memory'))

def consolidation(task_idx, old_net, cur_net, taskset, test_taskset, config, method_config, data_config, logger):
    
    model = config['model']
    cml_classes = config['cml_classes']
    batch_size = config['batch_size']
    classes_per_task = config['classes_per_task']
    task_path = config['task_path']
    num_workers = config['num_workers']
    device = config['device']

    print(model, cml_classes, batch_size, classes_per_task)

    process_list = method_config['consolidation_process_list']

    log = utils.Log(task_path)
    epoch = 0
    single_best_accuracy, multi_best_accuracy = 0.0, 0.0
    net = selector.model(model, device, cml_classes)

    for param in old_net.parameters():
        param.requires_grad = False

    for param in cur_net.parameters():
        param.requires_grad = False

    for process in process_list:
        epochs = process['epochs']
        optimizer = process['optimizer'](net.parameters())
        scheduler = process['scheduler'](optimizer)
        criterion = nn.MSELoss()

        train_loader = torch.utils.data.DataLoader(taskset, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_taskset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

        log.info("Start Consolidation")
        for ep in range(epochs):
            log.info("%d Epoch Started" % epoch)
            net.train()
            old_net.eval()
            cur_net.eval()
            epoch_loss = 0.0
            total = 0

            for i, data in enumerate(train_loader):
                utils.printProgressBar(i + 1, len(train_loader), prefix='train')
                images = data[0].to(device)
                cur_batch_size = images.size(0)

                optimizer.zero_grad()

                outputs_old = old_net(images)
                outputs_cur = cur_net(images)
                outputs_old -= outputs_old.mean(dim=1).reshape(cur_batch_size,-1)
                outputs_cur -= outputs_cur.mean(dim=1).reshape(cur_batch_size,-1)

                outputs_tot = torch.cat((outputs_old, outputs_cur), dim=1)
                outputs = net(images)
                loss = criterion(outputs, outputs_tot)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * cur_batch_size
                total += cur_batch_size

            epoch_loss /= total
            selector.scheduler.step(scheduler, epoch_loss)

            log.info("epoch: %d  train_loss: %.3lf  train_sample: %d" % (epoch, epoch_loss, total))

            if ep == (epochs - 1):
                test_loss, single_total_accuracy, single_class_accuracy, multi_total_accuracy, multi_class_accuracy = \
                    test(net, test_loader, config, data_config, task_idx, False)
                torch.save(net, os.path.join(task_path,'consolidated_model'))
            logger.epoch_step()
            epoch += 1

        log.info("Finish Consolidation")
    
    return net


def _train(task_idx, net, train_taskset, sample_memory, test_taskset, config, method_config, data_config, logger):
    """
    Args:
        config (dict): config file dictionary.
            task_path (str): directory for save. (taskwise, save_path + task**)
            cml_classes (int): size of cumulative taskset
    """
    batch_size = config['batch_size']
    classes_per_task = config['classes_per_task']
    task_path = config['task_path']
    num_workers = config['num_workers']
    device = config['device']
    cml_classes = config['cml_classes']

    process_list = method_config['process_list']

    log = utils.Log(task_path)
    epoch = 0
    single_best_accuracy, multi_best_accuracy = 0.0, 0.0
    for process in process_list:
        epochs = process['epochs']
        balance_finetune = process['balance_finetune']
        optimizer = process['optimizer'](net.parameters())
        scheduler = process['scheduler'](optimizer)
        criterion = nn.CrossEntropyLoss()

        if balance_finetune and cml_classes != classes_per_task:
            train_set = copy.deepcopy(sample_memory)
            train_set.update(BF=True)
        else:
            train_set = torch.utils.data.ConcatDataset((train_taskset, sample_memory))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_taskset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

        log.info("Start Training")
        for ep in range(epochs):
            log.info("%d Epoch Started" % epoch)
            net.train()
            epoch_loss = 0.0
            total = 0

            for i, data in enumerate(train_loader):
                utils.printProgressBar(i + 1, len(train_loader), prefix='train')
                images, labels = data[0].to(device), data[1].to(device)
                cur_batch_size = images.size(0)

                optimizer.zero_grad()

                outputs = net(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * cur_batch_size
                total += cur_batch_size

            epoch_loss /= total
            selector.scheduler.step(scheduler, epoch_loss)

            log.info("epoch: %d  train_loss: %.3lf  train_sample: %d" % (epoch, epoch_loss, total))
            if ep == (epochs - 1):
                test_loss, single_total_accuracy, single_class_accuracy = \
                    test(net, test_loader, config, data_config, task_idx, True)
            logger.epoch_step()
            epoch += 1

        make_heatmap(net, test_loader, config)
        log.info("Finish Training")
    
    return net
