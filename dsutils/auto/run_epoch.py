import torch
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, criterion, epoch, log_interval, logs):
    model.train()
    batch_size = train_loader.batch_size

    test_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(batch_size, -1)  # todo remove flatten

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # print(output.shape)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    train_acc = 100. * correct / len(train_loader.dataset)
    logs['train_acc'].append(train_acc)


def test(model, device, test_loader, criterion, logs):
    model.eval()
    batch_size = test_loader.batch_size

    test_loss = 0
    correct = 0
    actuals = []
    preds = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(batch_size, -1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            preds.append(output)
            actuals.append(target)

    log_act = torch.cat(actuals, 0).numpy()
    log_preds = torch.cat(preds, 0).numpy()

    logs['actual'] = log_act
    logs['predicted'] = log_preds

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    logs['test_acc'].append(test_acc)


# hacky garbo solution
def vae_train(model, device, train_loader, optimizer, criterion, epoch, log_interval):
    model.train()
    batch_size = train_loader.batch_size

    for batch_idx, (data, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(batch_size, -1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
