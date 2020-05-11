import torch
import os
import numpy as np
import torch.utils.data
import selector
import utils

from dataset import taskset


def get_single_total_accuracy(save_path, task_idx):
    state = torch.load(os.path.join(save_path, 'task%02d' % task_idx, 'LastModel.pth.tar'))
    return state['single_total_accuracy']


def get_single_class_accuracy(save_path, task_idx):
    state = torch.load(os.path.join(save_path, 'task%02d' % task_idx, 'LastModel.pth.tar'))
    return state['single_class_accuracy']


def get_single_task_accuracy(save_path, model_task_idx, task_idx, classes_per_task):
    state = torch.load(os.path.join(save_path, 'task%02d' % model_task_idx, 'LastModel.pth.tar'))
    return np.mean(state['single_class_accuracy'][task_idx * classes_per_task:(task_idx + 1) * classes_per_task])


def get_multi_total_accuracy(save_path, task_idx):
    state = torch.load(os.path.join(save_path, 'task%02d' % task_idx, 'LastModel.pth.tar'))
    return state['multi_total_accuracy']


def get_multi_class_accuracy(save_path, task_idx):
    state = torch.load(os.path.join(save_path, 'task%02d' % task_idx, 'LastModel.pth.tar'))
    return state['multi_class_accuracy']


def get_multi_task_accuracy(save_path, model_task_idx, task_idx, classes_per_task):
    state = torch.load(os.path.join(save_path, 'task%02d' % model_task_idx, 'LastModel.pth.tar'))
    return np.mean(state['multi_class_accuracy'][task_idx * classes_per_task:(task_idx + 1) * classes_per_task])


def get_e2e_accuracy(save_path, task_idx=10):
    cou = 0
    single_total_accuracy = 0.0
    multi_total_accuracy = 0.0

    for idx in range(1, task_idx):
        single_total_accuracy += get_single_total_accuracy(save_path, idx)
        multi_total_accuracy += get_multi_total_accuracy(save_path, idx)
        cou += 1

    single_e2e_accuracy = single_total_accuracy / cou
    multi_e2e_accuracy = multi_total_accuracy / cou

    return single_e2e_accuracy, multi_e2e_accuracy


def get_avg_accuracy(save_path, task_idx=10):
    cou = 0
    single_total_accuracy = 0.0
    multi_total_accuracy = 0.0

    for idx in range(task_idx):
        single_total_accuracy += get_single_total_accuracy(save_path, idx)
        multi_total_accuracy += get_multi_total_accuracy(save_path, idx)
        cou += 1

    single_avg_accuracy = single_total_accuracy / cou
    multi_avg_accuracy = multi_total_accuracy / cou

    return single_avg_accuracy, multi_avg_accuracy


def get_forgetting(save_path, task_idx, max_task_idx=10, classes_per_task=10):
    pre_single_accuracy = 0
    pre_multi_accuracy = 0
    for idx in range(task_idx, max_task_idx - 1):
        pre_single_accuracy = max(pre_single_accuracy,
                                  get_single_task_accuracy(save_path, idx, task_idx, classes_per_task))
        pre_multi_accuracy = max(pre_multi_accuracy,
                                 get_multi_task_accuracy(save_path, idx, task_idx, classes_per_task))

    cur_single_accuracy = get_single_task_accuracy(save_path, max_task_idx - 1, task_idx, classes_per_task)
    cur_multi_accuracy = get_multi_task_accuracy(save_path, max_task_idx - 1, task_idx, classes_per_task)

    single_forgetting = pre_single_accuracy - cur_single_accuracy
    multi_forgetting = pre_multi_accuracy - cur_multi_accuracy

    return single_forgetting, multi_forgetting


def get_avg_forgetting(save_path, task_idx=10, classes_per_task=10):
    cou = 0
    single_total_forgetting = 0.0
    multi_total_forgetting = 0.0
    for idx in range(task_idx - 1):
        forgetting = get_forgetting(save_path, idx, task_idx, classes_per_task)
        single_total_forgetting += forgetting[0]
        multi_total_forgetting += forgetting[1]
        cou += 1

    single_avg_forgetting = single_total_forgetting / cou
    multi_avg_forgetting = multi_total_forgetting / cou

    return single_avg_forgetting, multi_avg_forgetting


def get_avg_forgetting_list(save_path, max_task_idx=10, classes_per_task=10):
    forgetting_list = []
    for task_idx in range(2, max_task_idx + 1):
        cou = 0
        single_total_forgetting = 0.0
        multi_total_forgetting = 0.0
        for idx in range(task_idx - 1):
            forgetting = get_forgetting(save_path, idx, task_idx, classes_per_task)
            single_total_forgetting += forgetting[0]
            multi_total_forgetting += forgetting[1]
            cou += 1

        single_avg_forgetting = single_total_forgetting / cou
        multi_avg_forgetting = multi_total_forgetting / cou

        forgetting_list.append(single_avg_forgetting)

    return forgetting_list


def get_intransigence(save_path, ref_path, task_idx=10, classes_per_task=10):
    single_ref_accuracy = get_single_task_accuracy(ref_path, task_idx - 1, task_idx - 1, classes_per_task)
    multi_ref_accuracy = get_multi_task_accuracy(ref_path, task_idx - 1, task_idx - 1, classes_per_task)

    single_task_accuracy = get_single_task_accuracy(save_path, task_idx - 1, task_idx - 1, classes_per_task)
    multi_task_accuracy = get_multi_task_accuracy(save_path, task_idx - 1, task_idx - 1, classes_per_task)

    single_intransigence = single_ref_accuracy - single_task_accuracy
    multi_intransigence = multi_ref_accuracy - multi_task_accuracy

    return single_intransigence, multi_intransigence


def sample_dynamics(base_model, new_model, test_loader, base_cml_classes, new_cml_classes, device):
    base_model.eval()
    new_model.eval()
    both_true = 0
    forgetting = 0
    learn = 0
    both_false = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            cur_batch_size = images.size(0)

            base_outputs = base_model(images)[:, :base_cml_classes]
            new_outputs = new_model(images)[:, :new_cml_classes]

            _, base_predicted = torch.max(base_outputs, 1)
            _, new_predicted = torch.max(new_outputs, 1)

            base_correct = (base_predicted == labels)
            new_correct = (new_predicted == labels)

            both_true += torch.sum(base_correct & new_correct).item()
            forgetting += torch.sum(base_correct & ~new_correct).item()
            learn += torch.sum(~base_correct & new_correct).item()
            both_false += torch.sum(~base_correct & ~new_correct).item()

    return both_true, forgetting, learn, both_false


def get_sample_dynamics_forgetting(config, data_config):
    model = config['model']
    save_path = config['save_path']
    data_path = config['data_path']
    batch_size = config['batch_size']
    classes_per_task = config['classes_per_task']
    num_workers = config['num_workers']
    device = config['device']

    total_classes = data_config['total_classes']
    test_transform = data_config['transform']['test']
    curriculum = data_config['curriculums']
    curriculum = [curriculum[x:x + classes_per_task] for x in range(0, total_classes, classes_per_task)]

    cou = 0
    sum_forgetting = 0.0
    last_forgetting = 0.0

    test_task = []
    for task_idx, task in enumerate(curriculum):
        if task_idx == len(curriculum) - 1:
            break

        base_model = selector.model(model, device, total_classes)
        new_model = selector.model(model, device, total_classes)

        utils.load_checkpoint(base_model, os.path.join(save_path, 'task%02d' % task_idx, 'LastModel.pth.tar'))
        utils.load_checkpoint(new_model, os.path.join(save_path, 'task%02d' % (task_idx + 1), 'LastModel.pth.tar'))

        test_task.extend(task)
        test_taskset = taskset.Taskset(data_path, test_task, 0, train=False, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_taskset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

        base_cml_classes = (task_idx + 1) * classes_per_task
        new_cml_classes = (task_idx + 2) * classes_per_task

        sample_dynamics_value = sample_dynamics(base_model, new_model, test_loader, base_cml_classes, new_cml_classes,
                                                device)

        last_forgetting = sample_dynamics_value[1] / (sample_dynamics_value[0] + sample_dynamics_value[1])
        sum_forgetting += last_forgetting
        cou += 1

    avg_forgetting = sum_forgetting / cou
    return last_forgetting, avg_forgetting


def get_sample_dynamics_forgetting_list(config, data_config):
    model = config['model']
    save_path = config['save_path']
    data_path = config['data_path']
    batch_size = config['batch_size']
    classes_per_task = config['classes_per_task']
    num_workers = config['num_workers']
    device = config['device']

    total_classes = data_config['total_classes']
    test_transform = data_config['transform']['test']
    curriculum = data_config['curriculums']
    curriculum = [curriculum[x:x + classes_per_task] for x in range(0, total_classes, classes_per_task)]

    forgetting_list = []

    test_task = []
    for task_idx, task in enumerate(curriculum):
        if task_idx == len(curriculum) - 1:
            break

        base_model = selector.model(model, device, total_classes)
        new_model = selector.model(model, device, total_classes)

        utils.load_checkpoint(base_model, os.path.join(save_path, 'task%02d' % task_idx, 'LastModel.pth.tar'))
        utils.load_checkpoint(new_model, os.path.join(save_path, 'task%02d' % (task_idx + 1), 'LastModel.pth.tar'))

        test_task.extend(task)
        test_taskset = taskset.Taskset(data_path, test_task, 0, train=False, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_taskset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

        base_cml_classes = (task_idx + 1) * classes_per_task
        new_cml_classes = (task_idx + 2) * classes_per_task

        sample_dynamics_value = sample_dynamics(base_model, new_model, test_loader, base_cml_classes, new_cml_classes,
                                                device)

        forgetting_list.append(100 * (sample_dynamics_value[1] / (sample_dynamics_value[0] + sample_dynamics_value[1])))

    return forgetting_list


def get_sample_dynamics_intransigence(config, data_config):
    model = config['model']
    save_path = config['save_path']
    ref_path = config['ref_path']
    data_path = config['data_path']
    batch_size = config['batch_size']
    classes_per_task = config['classes_per_task']
    num_workers = config['num_workers']
    device = config['device']

    total_classes = data_config['total_classes']
    test_transform = data_config['transform']['test']
    curriculum = data_config['curriculums']
    curriculum = [curriculum[x:x + classes_per_task] for x in range(0, total_classes, classes_per_task)]

    cou = 0
    sum_intransigence = 0.0
    last_intransigence = 0.0

    test_task = []
    for task_idx, task in enumerate(curriculum):
        if task_idx == 0:
            continue

        base_model = selector.model(model, device, total_classes)
        new_model = selector.model(model, device, total_classes)

        utils.load_checkpoint(base_model, os.path.join(ref_path, 'task%02d' % task_idx, 'LastModel.pth.tar'))
        utils.load_checkpoint(new_model, os.path.join(save_path, 'task%02d' % task_idx, 'LastModel.pth.tar'))

        test_task = task
        test_taskset = taskset.Taskset(data_path, test_task, task_idx, train=False, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_taskset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

        base_cml_classes = (task_idx + 1) * classes_per_task
        new_cml_classes = (task_idx + 1) * classes_per_task

        sample_dynamics_value = sample_dynamics(base_model, new_model, test_loader, base_cml_classes, new_cml_classes,
                                                device)

        last_intransigence = sample_dynamics_value[1] / (sample_dynamics_value[0] + sample_dynamics_value[1])
        sum_intransigence += last_intransigence
        cou += 1

    avg_intransigence = sum_intransigence / cou
    return last_intransigence, avg_intransigence


def get_sample_dynamics_learn(config, data_config):
    model = config['model']
    save_path = config['save_path']
    data_path = config['data_path']
    batch_size = config['batch_size']
    classes_per_task = config['classes_per_task']
    num_workers = config['num_workers']
    device = config['device']

    total_classes = data_config['total_classes']
    test_transform = data_config['transform']['test']
    curriculum = data_config['curriculums']
    curriculum = [curriculum[x:x + classes_per_task] for x in range(0, total_classes, classes_per_task)]

    cou = 0
    sum_learn = 0.0
    last_learn = 0.0

    pre_task = []
    test_task = []
    for task_idx, task in enumerate(curriculum):
        if task_idx == 0:
            continue

        base_model = selector.model(model, device, total_classes)
        new_model = selector.model(model, device, total_classes)

        utils.load_checkpoint(base_model, os.path.join(save_path, 'task%02d' % (task_idx - 1), 'LastModel.pth.tar'))
        utils.load_checkpoint(new_model, os.path.join(save_path, 'task%02d' % task_idx, 'LastModel.pth.tar'))

        pre_task.extend(test_task)
        test_task = task

        test_taskset = taskset.Taskset(data_path, pre_task, 0, train=False, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_taskset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

        base_cml_classes = task_idx * classes_per_task
        new_cml_classes = (task_idx + 1) * classes_per_task

        pre_sample_dynamics_value = sample_dynamics(base_model, new_model, test_loader, base_cml_classes,
                                                    new_cml_classes, device)

        test_taskset = taskset.Taskset(data_path, test_task, 0, train=False, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_taskset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

        base_cml_classes = (task_idx + 1) * classes_per_task
        new_cml_classes = (task_idx + 1) * classes_per_task

        cur_sample_dynamics_value = sample_dynamics(base_model, new_model, test_loader, base_cml_classes,
                                                    new_cml_classes, device)

        last_learn = (pre_sample_dynamics_value[2] + cur_sample_dynamics_value[0] + cur_sample_dynamics_value[2]) / (
                pre_sample_dynamics_value[2] + pre_sample_dynamics_value[3] + len(test_taskset))
        sum_learn += last_learn
        cou += 1

    avg_learn = sum_learn / cou
    return last_learn, avg_learn


def get_sample_dynamics_penalty(config, data_config):
    model = config['model']
    save_path = config['save_path']
    ref_path = config['ref_path']
    data_path = config['data_path']
    batch_size = config['batch_size']
    classes_per_task = config['classes_per_task']
    num_workers = config['num_workers']
    device = config['device']

    total_classes = data_config['total_classes']
    test_transform = data_config['transform']['test']
    curriculum = data_config['curriculums']
    curriculum = [curriculum[x:x + classes_per_task] for x in range(0, total_classes, classes_per_task)]

    cou = 0
    sum_penalty = 0.0
    last_penalty = 0.0

    test_task = []
    for task_idx, task in enumerate(curriculum):
        if task_idx == 0:
            continue

        base_model = selector.model(model, device, total_classes)
        new_model = selector.model(model, device, total_classes)

        utils.load_checkpoint(base_model, os.path.join(ref_path, 'task%02d' % task_idx, 'LastModel.pth.tar'))
        utils.load_checkpoint(new_model, os.path.join(save_path, 'task%02d' % task_idx, 'LastModel.pth.tar'))

        test_task = task
        test_taskset = taskset.Taskset(data_path, test_task, task_idx, train=False, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_taskset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

        base_cml_classes = (task_idx + 1) * classes_per_task
        new_cml_classes = (task_idx + 1) * classes_per_task

        sample_dynamics_value = sample_dynamics(base_model, new_model, test_loader, base_cml_classes, new_cml_classes,
                                                device)

        last_penalty = sample_dynamics_value[2] / (sample_dynamics_value[2] + sample_dynamics_value[3])
        sum_penalty += last_penalty
        cou += 1

    avg_penalty = sum_penalty / cou
    return last_penalty, avg_penalty


def get_current_task_learn(save_path, max_task_idx=10, classes_per_task=10):
    cou = 0
    sum_current_task_learn = 0.0
    last_current_task_learn = 0.0
    for idx in range(1, max_task_idx):
        last_current_task_learn = get_single_task_accuracy(save_path, idx, idx, classes_per_task)
        sum_current_task_learn += last_current_task_learn
        cou += 1

    avg_current_task_learn = sum_current_task_learn / cou
    return last_current_task_learn, avg_current_task_learn


def get_fgt(save_path, max_task_idx=10, classes_per_task=10):
    pre_single_accuracy = 0
    pre_multi_accuracy = 0

    sum_single_accuracy = 0
    sum_multi_accuracy = 0
    for task_i in range(max_task_idx - 1):
        pre_single_accuracy = get_single_task_accuracy(save_path, task_i, task_i, classes_per_task)
        pre_multi_accuracy = get_multi_task_accuracy(save_path, task_i, task_i, classes_per_task)
        for task_j in range(task_i + 1, max_task_idx):
            sum_single_accuracy += (pre_single_accuracy - get_single_task_accuracy(save_path, task_j, task_i,
                                                                                   classes_per_task)) / (task_j + 1)
            sum_multi_accuracy += (pre_multi_accuracy - get_multi_task_accuracy(save_path, task_j, task_i,
                                                                                classes_per_task)) / (task_j + 1)
    single_fgt, multi_fgt = sum_single_accuracy / (max_task_idx - 1), sum_multi_accuracy / (max_task_idx - 1)
    return single_fgt, multi_fgt
