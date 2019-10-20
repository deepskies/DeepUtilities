import torch
import torch.nn.functional as F

def single_epoch(train_loader, config, epoch):
    model, optimizer, device, loss_fxn, epochs, print_freq = config.values()
    model.train()
    batch_size = train_loader.batch_size

    test_loss = 0
    correct = 0
    targets = []
    preds = []

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.to(device)
        data = data.view(batch_size, -1).to(device)

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
                print(f'epoch: {epoch}, batch: {batch_idx}, train_loss: {loss.item()}, acc: {batch_acc}/{batch_size}')

            targets.append(target)
            preds.append(pred)


# hacky garbo solution
def vae_train(train_loader, config, epoch):
    model, optimizer, device, loss_fxn, epochs, print_freq = config.values()
    model.train()
    batch_size = train_loader.batch_size
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.to(device)
        data = data.view(batch_size, -1).to(device)

        optimizer.zero_grad()
        enc, dec = model(data)

        # pred = enc.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        # batch_acc = pred.eq(target.view_as(pred)).sum().item()
        # correct += batch_acc
        
        loss = loss_fxn(dec, data)
        loss.backward()
        optimizer.step()
        if batch_idx % print_freq == 0:
            print(f'epoch: {epoch}, batch: {batch_idx}, loss: {loss}')
