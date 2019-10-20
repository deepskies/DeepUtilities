import torch
import torch.nn.functional as F


def test(test_loader, config, epoch):
    model, optimizer, device, loss_fxn, epochs, print_freq = config.values()

    model.eval()
    batch_size = test_loader.batch_size

    preds = []
    actuals = []

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.view(batch_size, -1)
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = loss_fxn(output, target).item() # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            batch_acc = pred.eq(target.view_as(pred)).sum().item()

            # print(target)
            if i % print_freq == 0:
                print(f'epoch: {epoch}, batch: {i}, test_loss: {loss}, acc: {batch_acc}/{batch_size}')

            preds += output.tolist()
            actuals += target.tolist()

    return preds, actuals

def vae_test(test_loader, config, epoch):
    model, optimizer, device, loss_fxn, epochs, print_freq = config.values()

    batch_size = test_loader.batch_size

    preds = []
    actuals = []
    decoded = []

    for i, (data, target) in enumerate(test_loader):
        data = data.view(batch_size, -1).to(device)
        target = target.to(device)

        enc, dec = model(data)

        pred = enc.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        batch_acc = pred.eq(target.view_as(pred)).sum().item()  # THIS SHOULD BE NOISE RN

        batch_loss = loss_fxn(dec, data).item()

        if i % print_freq == 0:
            print(f'epoch: {epoch}, batch: {i}, test_loss: {batch_loss}, acc: {batch_acc}/{batch_size}')

        preds += enc.tolist()
        actuals += target.tolist()
        decoded += dec.tolist()

    return preds, actuals, decoded
