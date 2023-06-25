import torch
from progressbar import ProgressBar, Percentage, Bar, ETA

from lib.utils import save_state

SAVEPATH = "checkpoints"
SAVEPATTERN = "model_{}.pth"

def train_step(model,
               dataloader,
               loss_fn,
               optimizer,
               device):

    widgets = ['Train step over batches: ', Percentage(), ' ', Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=1e4).start()


    # model to train mode
    model.train()

    _loss, _acc = 0, 0

    # Loop through data loader data batches
    for batch, (_input, _output) in pbar(enumerate(dataloader)):
        # Send data to target device
        _input, _output = _input.to(device), _output.to(device)

        # forward pass
        _output_logits = model(_input)

        loss = loss_fn(_output_logits, _output)
        _loss += loss.item()

        # need to initialize gradients to zero
        optimizer.zero_grad()

        # calculate jacobian
        loss.backward()

        # one step descent
        optimizer.step()

        # accuracy metric across all batches
        _output_logits_class = _output_logits.argmax(dim=1)
        _output_true_class = _output.argmax(dim=1)
        _acc += (_output_logits_class == _output_true_class).sum().item()/len(_output_logits)

    # average loss and accuracy per batch
    _loss = _loss / len(dataloader)
    _acc = _acc / len(dataloader)
    return _loss, _acc

def test_step(model,
              dataloader,
              loss_fn,
              optimizer,
              device):


    widgets = ['Test step over batches: ', Percentage(), ' ', Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=1e4).start()


    # set to evaluation mode, no dropout / batchnorm
    model.eval()

    _loss, _acc = 0, 0

    # inference context manager
    with torch.inference_mode():

        for batch, (_input, _output) in pbar(enumerate(dataloader)):
            # Send data to target device
            _input , _output = _input.to(device), _output.to(device)

            # prediction
            _pred_logits = model(_input)

            loss = loss_fn(_pred_logits, _output)
            _loss += loss.item()

            # accuracy
            _pred_labels = _pred_logits.argmax(dim=1)
            _output_true_class = _output.argmax(dim=1)
            _acc += ((_pred_labels == _output_true_class).sum().item()/len(_pred_labels))

    _loss = _loss / len(dataloader)
    _acc = _acc / len(dataloader)
    return _loss, _acc

def train_loop(model,
               train_loader,
               test_loader,
               optimizer,
               loss_fn,
               epochs,
               device,
               checkpoint=2,
               ):


    # to follow progress over epochs
    widgets = ['Epoch: ', Percentage(), ' ', Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=1e4).start()

    # send model to chosen device
    model.to(device)

    # store results for later plotting

    results = {
        "training_loss" : [],
        "testing_loss"  : [],
        "training_acc"  : [],
        "testing_acc"   : []
    }


    for i in pbar(range(epochs)):

        train_loss, train_acc = train_step(model,
                                           train_loader,
                                           loss_fn,
                                           optimizer,
                                           device)

        test_loss, test_acc = test_step(model,
                                        test_loader,
                                        loss_fn,
                                        optimizer,
                                        device)

        print(f"Training -- loss: {train_loss:.2e}, acc: {train_acc:.2e}")
        print(f"Training -- loss: {test_loss:.2e}, acc: {test_acc:.2e}")

        results["training_loss"].append(train_loss)
        results["training_acc"].append(train_acc)
        results["testing_loss"].append(test_loss)
        results["testing_acc"].append(test_acc)

        if i%checkpoint:
            save_state(model,
                       SAVEPATH,
                       SAVEPATTERN.format(i))

    return results
