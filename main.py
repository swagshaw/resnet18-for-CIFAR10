import neptune.new as neptune
import torch.nn as nn
from test import test
import torch
from ResNet import ResNet18, testloader, trainloader
import torch.optim as optim
import argparse


# Training
def train(epoch, net, criterion, trainloader, scheduler,flag):
    device = 'cuda'
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print("iteration : %3d, loss : %0.4f, accuracy : %2.2f" % (
                batch_idx + 1, train_loss / (batch_idx + 1), 100. * correct / total))
    if flag:
        scheduler.step()
    return train_loss / (batch_idx + 1), 100. * correct / total


def get_args():
    """
    Get arguments from the input command line.
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', help='weight_decay', type=float, default=0,
                        required=False)
    parser.add_argument('-s', type=bool, default=False, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    parameters = get_args()
    # main body
    config = {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': parameters.w,
        'scheduler': parameters.s
    }

    # initialize Neptune
    run = neptune.init(project='swagshaw/AI6103-Assignment',
                       api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0'
                                 'cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2OWU'
                                 'zZjNmZS0yYTdlLTQyNjctYWE2Mi0zODI1MTA0NzI1ZTIifQ==')
    run["parameters"] = config

    net = ResNet18().to('cuda')
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = optim.SGD(net.parameters(), lr=config['lr'],
                          momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(1, 301):
        train_loss, train_acc = train(epoch, net, criterion, trainloader, scheduler,flag=parameters.s)
        test_loss, test_acc = test(epoch, net, criterion, testloader)
        run['train/epoch/loss'].log(train_loss)
        run['train/epoch/accuracy'].log(train_acc)
        run['test/epoch/loss'].log(test_loss)
        run['test/epoch/accuracy'].log(test_acc)
        print(("Epoch : %3d, training loss : %0.4f, training accuracy : %2.2f, test loss " + \
               ": %0.4f, test accuracy : %2.2f") % (epoch, train_loss, train_acc, test_loss, test_acc))
