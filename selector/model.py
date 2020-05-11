import net
import torch.nn as nn


def model(model, device, num_classes):
    model = getattr(net, model)(num_classes=num_classes)
    model.to(device)
    #model = nn.DataParallel(model)

    return model
