import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import compress


def clamp(value, x, y):
    return min(max(x, value), y)


def bright(image):
    value = random.randint(-63, 63)
    pixels = image.load()
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            pixels[i, j] = (clamp(pixels[i, j][0] + value, 0, 255),
                            clamp(pixels[i, j][1] + value, 0, 255),
                            clamp(pixels[i, j][2] + value, 0, 255))
    return image


def get_good_filenames(score_set, k, padding, stride, mode='intuitive'):
    filenames = []
    labels = torch.Tensor(score_set.targets)
    threshold = 1000

    if mode == 'fish':
        sum_max = 0.0
        for i in range(len(score_set.task)):
            max_max = torch.max(score_set.scores[labels == score_set.task[i]].detach().cpu())
            sum_max += max_max
        threshold = (sum_max / len(score_set.task)).item()

    for i in score_set.task:
        score = score_set.scores[labels == i]

        if mode == 'new':
            start = 0
            plt.clf()
            '''
            n, bins, patches = plt.hist(score, bins=50, range=(0, 100))
            score = compose_score(n / sum(n), bins, score)
            files = list(compress(score_set.filenames, list(np.array(labels == i))))
            sor_files = [file for _, file in sorted(zip(score, files), reverse=True)]
            choices = np.random.choice(stride * k if stride*k<len(sor_files) else len(sor_files), k)
            filenames.append([sor_files[start + ele] for ele in choices])
            '''

        elif mode == 'intuitive':
            files = list(compress(score_set.filenames, list(np.array(labels == i))))
            sor_files = [file for _, file in sorted(zip(score, files), reverse=True)]
            start = padding
            choices = np.random.choice(min(stride * k, 400), k, replace=False).tolist()
            choices.sort()
            # filenames.append(sor_files[start:start + stride * k:stride])
            filenames.append([sor_files[start + ele] for ele in choices])

        elif mode == 'fish':
            files = list(compress(score_set.filenames, list(np.array(labels == i))))
            sor_files = [file for _, file in sorted(zip(score, files), reverse=True)]
            sc_arr = np.array(sorted(score, reverse=True))
            start = np.sum(np.array(sc_arr > threshold, dtype=int))
            filenames.append(sor_files[start:start + k * stride:stride])

        else:
            print('no such scoring method')

    print('filenames', sum([len(filenames[i]) for i in range(len(filenames))]))
    return filenames


def fucking_scoring_samples(score_model, scoreset, batch_size, num_workers, device, classes_per_task):
    scoreloader = torch.utils.data.DataLoader(scoreset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    scores_in_cls = []
    embedding_in_cls = []

    for i in range(classes_per_task):
        embedding_in_cls.append([])
        scores_in_cls.append([])

    for idx, (data, label, indices) in enumerate(scoreloader):
        data, label = data.to(device), label.to(device)
        with torch.no_grad():
            feature = score_model.module.feature_out(data)
            for j in range(classes_per_task):
                embedding_in_cls[j].append(feature[label == j].detach().cpu())

    for j in range(classes_per_task):
        embedding_in_cls[j] = torch.cat([embedding_in_cls[j][i] for i in range(len(embedding_in_cls[j]))], dim=0)

    mean_in_cls = []
    for j in range(classes_per_task):
        mean_in_cls.append(torch.mean(embedding_in_cls[j], 0).unsqueeze(dim=0))
    mean_vectors = torch.cat(mean_in_cls, dim=0)
    data_embedded_mean = torch.mean(mean_vectors, dim=0)

    for idx, (data, label, indices) in enumerate(scoreloader):
        data, label = data.to(device), label.to(device)
        with torch.no_grad():
            feature = score_model.module.feature_out(data)
            score = torch.sqrt(torch.sum(
                (torch.cat([data_embedded_mean.unsqueeze(0)] * feature.size(0), dim=0) - feature.detach().cpu()) ** 2,
                dim=1))
            # print(score.size())
            scoreset.update_score(indices, score.detach().cpu())


def scoring_samples(net, scoreset, batch_size, num_workers, device, cml_classes):
    score_loader = torch.utils.data.DataLoader(scoreset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    softmax = nn.Softmax(dim=1)

    for i, data in enumerate(score_loader):
        net.eval()
        with torch.no_grad():
            images, labels = data[0].to(device), data[1].to(device)
            indices = data[2].to(device)
            outputs = net(images)
            outputs = outputs[:, :cml_classes]
            prob = softmax(outputs)
            score = prob[[i for i in range(outputs.size(0))], labels]
            scoreset.update_score(indices, score.detach().cpu())


def printProgressBar(iteration, total, prefix='Progress: ', suffix='Complete', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    if iteration > total:
        total = iteration
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()


def save_checkpoint(state, save_path, filename):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)


def save_model(net, single_best_accuracy, multi_best_accuracy, single_total_accuracy, single_class_accuracy,
               multi_total_accuracy, multi_class_accuracy, filepath, cur_epoch, epochs):
    """
    Args:
        net (nn.Module): network will be saved
        single_best_accuracy (float): best accuracy of single-head (best of total_accuracy)
        multi_best_accuracy (float): best accuracy of multi-head (best of total_accuracy)
        single_total_accuracy (float): single-head total_accuracy
        single_class_accuracy (list of float): single-head accuracy for each class
        multi_total_accuracy (float): multi-head total_accuracy
        multi_class_accuracy (list of float): multi-head accuracy for each class
        filepath (str): where the net have to be saved
        cur_epoch (int): current epoch
        epochs (int): total number of epochs for the task
    """

    state = {
        'epoch': cur_epoch + 1,
        'state_dict': net.state_dict(),
        'single_total_accuracy': single_total_accuracy,
        'single_class_accuracy': single_class_accuracy,
        'multi_total_accuracy': multi_total_accuracy,
        'multi_class_accuracy': multi_class_accuracy,
    }
    # if it is the single-head best Model, we save it as "SingleBestModel.pth.tar"
    if cur_epoch >= 0 and single_best_accuracy < single_total_accuracy:
        single_best_accuracy = single_total_accuracy
        save_checkpoint(state, filepath, "SingleBestModel.pth.tar")

    # if it is the multi-head best Model, we save it as "MultiBestModel.pth.tar"
    if cur_epoch >= 0 and multi_best_accuracy < multi_total_accuracy:
        multi_best_accuracy = multi_total_accuracy
        save_checkpoint(state, filepath, "MultiBestModel.pth.tar")

    # if it is the last Model, we save it as "LastModel.pth.tar"
    if cur_epoch + 1 == epochs:
        idx = 0
        while os.path.exists(os.path.join(filepath, str(idx) + '.pth.tar')):
            idx += 1
        save_checkpoint(state, filepath, str(idx) + '.pth.tar')
        shutil.copyfile(os.path.join(filepath, str(idx) + '.pth.tar'), os.path.join(filepath, 'LastModel.pth.tar'))

    return single_best_accuracy, multi_best_accuracy


# Loads the saved state to be used for training.
def load_checkpoint(model, filepath):
    """
    Args:
        model (nn.Module): model that will load a checkpoint
        filepath (str): the path that checkpoint is on
    """
    state = torch.load(filepath)
    model.load_state_dict(state['state_dict'])
    del state
    torch.cuda.empty_cache()


class Log(object):
    def __init__(self, root):
        if not os.path.isdir(root):
            os.makedirs(root)
        self.root = os.path.join(root, 'log.log')

    def info(self, string):
        print(string)
        with open(self.root, 'a') as f:
            f.write(string + '\n')


def save_pyplot(save_path, classes_per_task, total_classes):
    idx = 0
    ys = {
        'single_best': [],
        'single_last': [],
        'multi_best': [],
        'multi_last': [],
    }
    while os.path.exists(os.path.join(save_path, 'task%02d' % idx)):
        ys['single_best'].append(
            torch.load(os.path.join(save_path, 'task%02d' % idx, 'SingleBestModel.pth.tar'))['single_total_accuracy'])
        ys['single_last'].append(
            torch.load(os.path.join(save_path, 'task%02d' % idx, 'LastModel.pth.tar'))['single_total_accuracy'])
        ys['multi_best'].append(
            torch.load(os.path.join(save_path, 'task%02d' % idx, 'MultiBestModel.pth.tar'))['multi_total_accuracy'])
        ys['multi_last'].append(
            torch.load(os.path.join(save_path, 'task%02d' % idx, 'LastModel.pth.tar'))['multi_total_accuracy'])
        idx += 1

    x = [t * classes_per_task for t in range(1, (total_classes // classes_per_task) + 1)]

    for key in ys.keys():
        y = ys[key]
        fig = plt.figure()
        plt.ylabel('Accuracy')
        plt.xlabel('Number of classes')
        plt.xlim(0, total_classes)
        plt.ylim(0, 100)
        plt.yticks([t * 10 for t in range(11)])
        plt.grid(True, axis='y')

        plt.plot(x, y, marker='.', markersize=9, label=save_path.split('/')[-2:] + [key])

        plt.legend(loc='lower left')

        for i, j in zip(x, y):
            plt.annotate('%.3lf' % j, xy=(i, j))

        fig.savefig(os.path.join(save_path, key + '.png'))

        plt.close(fig)


def save_heatmap(task_path, save_path, class_heatmap, task_heatmap):
    classes_per_task = len(class_heatmap) // len(task_heatmap)
    titles = ['Class HeatMap', 'Task HeatMap']
    file_names = ['class_heatmap', 'task_heatmap']
    heatmaps = [class_heatmap, task_heatmap]
    for title, file_name, heatmap in zip(titles, file_names, heatmaps):
        fig, ax = plt.subplots()
        ax.imshow(heatmap)

        if title == 'Task HeatMap':
            ax.set_xticks(np.arange(len(heatmap)))
            ax.set_yticks(np.arange(len(heatmap)))

            for i in range(len(heatmap[0])):
                for j in range(len(heatmap[0])):
                    ax.text(j, i, heatmap[i, j], ha='center', va='center', color='w')
        else:
            ax.set_xticks(np.arange(0, len(heatmap), classes_per_task))
            ax.set_yticks(np.arange(0, len(heatmap), classes_per_task))

        ax.set_title(title)
        fig.tight_layout()

        fig.savefig(os.path.join(task_path, file_name + '.png'))
        fig.savefig(os.path.join(save_path, file_name + '.png'))
        plt.close(fig)


def get_logit(net, data_loader, device, cml_classes, classes_per_task, log):
    logits = {}
    net.eval()
    with torch.no_grad():
        pre_sample_pre_logit_max = torch.tensor(0, dtype=torch.float32)
        pre_sample_pre_logit_mean_except_label = torch.tensor(0, dtype=torch.float32)
        pre_sample_cur_logit_max = torch.tensor(0, dtype=torch.float32)
        pre_sample_cur_logit_mean_except_label = torch.tensor(0, dtype=torch.float32)
        cur_sample_pre_logit_max = torch.tensor(0, dtype=torch.float32)
        cur_sample_pre_logit_mean_except_label = torch.tensor(0, dtype=torch.float32)
        cur_sample_cur_logit_max = torch.tensor(0, dtype=torch.float32)
        cur_sample_cur_logit_mean_except_label = torch.tensor(0, dtype=torch.float32)

        pre_sample_pre_logit_max_cou = torch.tensor(0, dtype=torch.float32)
        pre_sample_pre_logit_mean_except_label_cou = torch.tensor(0, dtype=torch.float32)
        pre_sample_cur_logit_max_cou = torch.tensor(0, dtype=torch.float32)
        pre_sample_cur_logit_mean_except_label_cou = torch.tensor(0, dtype=torch.float32)
        cur_sample_pre_logit_max_cou = torch.tensor(0, dtype=torch.float32)
        cur_sample_pre_logit_mean_except_label_cou = torch.tensor(0, dtype=torch.float32)
        cur_sample_cur_logit_max_cou = torch.tensor(0, dtype=torch.float32)
        cur_sample_cur_logit_mean_except_label_cou = torch.tensor(0, dtype=torch.float32)

        for i, data in enumerate(data_loader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)
            outputs = outputs[:, :cml_classes]
            for idx in range(outputs.size(0)):
                label = labels[idx].item()
                pre_logit = outputs[idx][:-classes_per_task]
                cur_logit = outputs[idx][-classes_per_task:]

                if label < cml_classes - classes_per_task:  # previous task
                    pre_sample_pre_logit_max += torch.max(pre_logit, 0)[0]
                    pre_sample_pre_logit_mean_except_label += torch.sum(
                        torch.cat([pre_logit[:label], pre_logit[label + 1:]]))
                    pre_sample_cur_logit_max += torch.max(cur_logit, 0)[0]
                    pre_sample_cur_logit_mean_except_label += torch.sum(cur_logit)

                    pre_sample_pre_logit_max_cou += 1
                    pre_sample_pre_logit_mean_except_label_cou += pre_logit.size(0) - 1
                    pre_sample_cur_logit_max_cou += 1
                    pre_sample_cur_logit_mean_except_label_cou += cur_logit.size(0)
                else:  # current task
                    label %= classes_per_task
                    if cml_classes != classes_per_task:
                        cur_sample_pre_logit_max += torch.max(pre_logit, 0)[0]
                        cur_sample_pre_logit_mean_except_label += torch.sum(pre_logit)
                    cur_sample_cur_logit_max += torch.max(cur_logit, 0)[0]
                    cur_sample_cur_logit_mean_except_label += torch.sum(
                        torch.cat([cur_logit[:label], cur_logit[label + 1:]]))

                    if cml_classes != classes_per_task:
                        cur_sample_pre_logit_max_cou += 1
                        cur_sample_pre_logit_mean_except_label_cou += pre_logit.size(0)
                    cur_sample_cur_logit_max_cou += 1
                    cur_sample_cur_logit_mean_except_label_cou += cur_logit.size(0) - 1

        pre_sample_pre_logit_max /= pre_sample_pre_logit_max_cou
        pre_sample_pre_logit_mean_except_label /= pre_sample_pre_logit_mean_except_label_cou
        pre_sample_cur_logit_max /= pre_sample_cur_logit_max_cou
        pre_sample_cur_logit_mean_except_label /= pre_sample_cur_logit_mean_except_label_cou
        cur_sample_pre_logit_max /= cur_sample_pre_logit_max_cou
        cur_sample_pre_logit_mean_except_label /= cur_sample_pre_logit_mean_except_label_cou
        cur_sample_cur_logit_max /= cur_sample_cur_logit_max_cou
        cur_sample_cur_logit_mean_except_label /= cur_sample_cur_logit_mean_except_label_cou

        logits['pre_sample_pre_logit_max'] = pre_sample_pre_logit_max
        logits['pre_sample_pre_logit_mean_except_label'] = pre_sample_pre_logit_mean_except_label
        logits['pre_sample_cur_logit_max'] = pre_sample_cur_logit_max
        logits['pre_sample_cur_logit_mean_except_label'] = pre_sample_cur_logit_mean_except_label
        logits['cur_sample_pre_logit_max'] = cur_sample_pre_logit_max
        logits['cur_sample_pre_logit_mean_except_label'] = cur_sample_pre_logit_mean_except_label
        logits['cur_sample_cur_logit_max'] = cur_sample_cur_logit_max
        logits['cur_sample_cur_logit_mean_except_label'] = cur_sample_cur_logit_mean_except_label

        log.info('pre_sample_pre_logit_max: %.5lf  pre_sample_pre_logit_mean_except_label: %.5lf'
                 % (pre_sample_pre_logit_max, pre_sample_pre_logit_mean_except_label))
        log.info('pre_sample_cur_logit_max: %.5lf  pre_sample_cur_logit_mean_except_label: %.5lf'
                 % (pre_sample_cur_logit_max, pre_sample_cur_logit_mean_except_label))
        log.info('cur_sample_pre_logit_max: %.5lf  cur_sample_pre_logit_mean_except_label: %.5lf'
                 % (cur_sample_pre_logit_max, cur_sample_pre_logit_mean_except_label))
        log.info('cur_sample_cur_logit_max: %.5lf  cur_sample_cur_logit_mean_except_label: %.5lf'
                 % (cur_sample_cur_logit_max, cur_sample_cur_logit_mean_except_label))

    return logits


def save_reliability_diagram(net, data_loader, device, cml_classes):
    net.eval()

    logit_list = [[] for _ in range(11)]
    correct_list = [0 for _ in range(11)]
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            cur_batch_size = images.size(0)

            outputs = net(images)[:, :cml_classes]
            soft_outputs = F.softmax(outputs, dim=1)

            max_value, predicted = torch.max(soft_outputs, 1)
            max_value *= 100.0

            corrects = (predicted == labels)

            for idx in range(cur_batch_size):
                cur_list_idx = int(max_value[idx].item() / 10)

                logit_list[cur_list_idx].append(max_value[idx].item())
                correct_list[cur_list_idx] += corrects[idx].item()

    correct_list[9] += correct_list[10]
    del correct_list[10]
    logit_list[9].extend(logit_list[10])
    del logit_list[10]

    accuracy = [(0.0 if len(logit) == 0 else 100.0 * (correct / len(logit))) for correct, logit in zip(correct_list, logit_list)]
    logit_mean = [(0.0 if np.isnan(np.mean(logit)) else np.mean(logit)) for logit in logit_list]
    cou = [len(logit) for logit in logit_list]
    gap = [logit - acc for acc, logit in zip(accuracy, logit_mean)]
    ece = np.sum([cur_cou * abs(cur_gap) for cur_cou, cur_gap in zip(cou, gap)]) / np.sum(cou)

    return {'accuracy': accuracy, 'logit_mean': logit_mean, 'cou': cou, 'gap': gap, 'ece': ece}
