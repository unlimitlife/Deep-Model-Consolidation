import numpy as np
import torch.nn as nn
import torch
import utils


def test(net, test_loader, config, data_config, task_idx, single = False):
    device = config['device']
    task_path = config['task_path']
    cml_classes = config['cml_classes']

    classes = data_config['classes']
    classes_per_task = config['classes_per_task']

    log = utils.Log(task_path)

    net.eval()
    total, single_total_correct, multi_total_correct, test_loss = 0.0, 0.0, 0.0, 0.0

    class_per_sample = np.zeros([cml_classes])

    single_class_correct = np.zeros([cml_classes])
    single_class_accuracy = np.zeros([cml_classes])

    multi_class_correct = np.zeros([cml_classes])
    multi_class_accuracy = np.zeros([cml_classes])

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            cur_batch_size = images.size(0)

            outputs = net(images)
            outputs = outputs[:, :cml_classes]

            loss = criterion(outputs, labels)
            test_loss += loss.item() * cur_batch_size

            _, predicted = torch.max(outputs, 1)
            if not single:
                multi_predicted = [torch.max(outputs[:, idx: idx + classes_per_task], 1)[1]
                               for idx in range(0, cml_classes, classes_per_task)]
            c = (predicted == labels)
            for idx in range(c.size()[0]):
                label = labels[idx]
                single_class_correct[label] += c[idx].item()
                if not single:
                    multi_head_task = label.item() // classes_per_task
                    multi_head_label = label.item() % classes_per_task
                    multi_class_correct[label] += (multi_predicted[multi_head_task][idx] == multi_head_label).item()
                class_per_sample[label] += 1

    range_class = range(classes_per_task) if single else range(cml_classes)
    for idx in range_class:
        total += class_per_sample[idx]

        single_total_correct += single_class_correct[idx]
        single_class_accuracy[idx] = 100.0 * single_class_correct[idx] / class_per_sample[idx]

        if not single:
            multi_total_correct += multi_class_correct[idx]
            multi_class_accuracy[idx] = 100.0 * multi_class_correct[idx] / class_per_sample[idx]
        
            log.info('Accuracy of %s - single-head: %.3lf%% / multi-head: %.3lf%%' %
                    (classes[idx], single_class_accuracy[idx], multi_class_accuracy[idx]))
        else:
            log.info('Accuracy of %s - single-head: %.3lf%%' %
                    (classes[idx+task_idx*classes_per_task], single_class_accuracy[idx]))

    test_loss /= total
    single_total_accuracy = 100.0 * single_total_correct / total
    if single:
        log.info("single-head test accuracy: %.3lf%% test_loss: %.3lf test_sample: %d" %
                (single_total_accuracy, test_loss, total))
        return test_loss, single_total_accuracy, single_class_accuracy
    else:
        multi_total_accuracy = 100.0 * multi_total_correct / total
        log.info("single-head test accuracy: %.3lf%% multi-head test accuracy: %.3lf%% test_loss: %.3lf test_sample: %d" %
                (single_total_accuracy, multi_total_accuracy, test_loss, total))
        return test_loss, single_total_accuracy, single_class_accuracy, multi_total_accuracy, multi_class_accuracy




def make_heatmap(net, test_loader, config):
    device = config['device']
    save_path = config['save_path']
    task_path = config['task_path']
    cml_classes = config['cml_classes']
    classes_per_task = config['classes_per_task']

    net.eval()

    class_heatmap = np.zeros((cml_classes, cml_classes), dtype=int)
    task_heatmap = np.zeros((cml_classes // classes_per_task, cml_classes // classes_per_task), dtype=int)

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            cur_batch_size = images.size(0)

            outputs = net(images)
            outputs = outputs[:, :cml_classes]

            _, predicted = torch.max(outputs, 1)

            for idx in range(cur_batch_size):
                label = labels[idx]
                predict = predicted[idx]
                class_heatmap[label][predict] += 1
                task_heatmap[label // classes_per_task][predict // classes_per_task] += 1

    utils.save_heatmap(task_path, save_path, class_heatmap, task_heatmap)
