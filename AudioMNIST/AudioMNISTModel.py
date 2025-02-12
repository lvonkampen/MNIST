import torch
import torch.nn as nn
import torch.nn.init as init


class AudioMnistModel(nn.Module):
    def __init__(self, activation, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

        self.activation = activation

        init.xavier_normal_(self.layer1.weight)
        init.xavier_normal_(self.layer2.weight)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.layer1(xb)
        out = self.activation(out)
        out = self.layer2(out)
        out = self.activation(out)
        return out

    def training_step(self, batch, loss_func):
        images, labels = batch
        out = self(images)  ## Generate predictions
        loss = loss_func(out, labels)  ## Calculate the loss
        return loss

    def train_epoch(self, train_loader, optim, loss_func):
        self.train()
        train_loss = 0

        for batch in train_loader:
            images, labels = batch
            optim.zero_grad()
            loss = self.training_step((images, labels), loss_func)
            loss.backward()
            optim.step()

            train_loss += loss.item()

        return train_loss / len(train_loader)

    def validation_step(self, batch, loss_func):
        images, labels = batch
        out = self(images)
        loss = loss_func(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validate_epoch(self, val_loader, loss_func):
        self.eval()
        correct = 0
        total = 0

        val_results = []
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                val_results.append(self.validation_step((images, labels), loss_func))

                _, preds = torch.max(self(images), dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accu = correct / total * 100
        return self.validation_epoch_end(val_results), accu

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.tensor(torch.sum(preds == labels).item()/ len(preds))
