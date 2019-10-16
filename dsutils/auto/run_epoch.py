import torch
import torch.nn.functional as F

def train(train_loader, optimizer, config):
    model, device, loss_fxn, epoch, print_freq, logs = config.values()
    model.train()
    batch_size = train_loader.batch_size

    test_loss = 0
    correct = 0
    targets = []
    preds = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(batch_size, -1)  # todo remove flatten

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fxn(output, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # print(output.shape)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            batch_acc = pred.eq(target.view_as(pred)).sum().item()
            correct += batch_acc

            if batch_idx % print_freq == 0:
                print(f'epoch: {epoch}, batch: {batch_idx}, train_loss: {loss.item()}, acc: {batch_acc}, batch_size: {batch_size}')

            targets.append(target)
            preds.append(pred)


    train_acc = 100. * correct / len(train_loader.dataset)
    logs['train_acc'].append(train_acc)
    # return preds, targets

def test(test_loader, config):
    model, device, loss_fxn, epoch, print_freq, logs = config.values()
    model.eval()
    batch_size = test_loader.batch_size

    test_loss = 0
    correct = 0
    actuals = []
    preds = []

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.view(batch_size, -1)

            data, target = data.to(device), target.to(device)

            output = model(data)
            batch_loss = loss_fxn(output, target).item() # sum up batch loss
            test_loss += batch_loss

            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            batch_acc = pred.eq(target.view_as(pred)).sum().item()
            correct += batch_acc
            # print(target)
            if i % print_freq == 0:
                print(f'epoch: {epoch}, batch: {i}, test_loss: {batch_loss}, acc: {batch_acc}/{batch_size}')


            preds += output.tolist()
            actuals += target.tolist()

    # log_act = torch.cat(actuals, 0).numpy()
    # log_preds = torch.cat(preds, 0).numpy()
    # print(actuals)
    # print(preds)
    # print(log_act)
    logs['actual'] += actuals
    logs['predicted'] += preds

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    logs['test_acc'].append(test_acc)



# hacky garbo solution
def vae_train(train_loader, optimizer, config):
    model, device, loss_fxn, epoch, print_freq, logs = config.values()
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
        if batch_idx % print_freq == 0:
            print(f'epoch: {epoch}, batch: {batch_idx}, loss: {loss}, batch_acc: {batch_acc}/{batch_size}')
