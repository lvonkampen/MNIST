import torch
import torch.nn as nn
import torch.nn.init as init

from Config import Hyperparameters


class ConvBlock(nn.Module):
    def __init__(self, activation, in_channels, out_channels, kernel_size=3, pool_size=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.bn   = nn.BatchNorm1d(out_channels)
        self.act  = activation
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.pool(x)

class WhisperVIAModel(nn.Module):
    def __init__(self, activation, hidden_sizes, in_channels, num_classes, conv_channels):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        c = in_channels
        for oc in conv_channels:
            self.conv_blocks.append(ConvBlock(activation, c, oc))
            c = oc
        self.adaptive_pool = nn.AdaptiveAvgPool1d(25) # this should be a parameter

        # compute flattened size
        with torch.no_grad(): # this can be computed from parameters
            dummy = torch.zeros(1, in_channels, 100)
            for b in self.conv_blocks: dummy = b(dummy)
            dummy = self.adaptive_pool(dummy)
            flat_size = dummy.view(1, -1).size(1)

        # fully‐connected
        layers = []
        prev = flat_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), activation]
            prev = h
        self.fc = nn.Sequential(*layers)
        self.out = nn.Linear(prev, num_classes) # There is no Sigmoid activation at the very end that stops values from going beyond 1.0 ?

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight); init.zeros_(m.bias)

    def forward(self, x):
        for b in self.conv_blocks:
            x = b(x)
        x = self.adaptive_pool(x).flatten(1)
        x = self.fc(x)
        return self.out(x)

    def training_step(self, batch, loss_func):
        images, labels = batch
        images, labels = images.to(Hyperparameters.device), labels.to(Hyperparameters.device)
        labels = labels.view(-1, 1).float()
        out = self(images)
        loss = loss_func(out, labels)
        return loss

    def train_epoch(self, data_loader, optimizer, loss_fn, device):
        self.train()
        total_loss = 0

        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = self(xb)
            loss = loss_fn(out.squeeze(), yb.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader) # len(data_loader) is batch size NOT segment
        return avg_loss

    def validation_step(self, batch, loss_func):
        images, labels = batch
        images, labels = images.to(Hyperparameters.device), labels.to(Hyperparameters.device)
        labels = labels.view(-1, 1).float()

        out = self(images)
        loss = loss_func(out, labels)
        return {'val_loss': torch.tensor(loss.item())}

    def validate_epoch(self, data_loader, loss_fn, device):
        self.eval()
        total_loss = 0

        with torch.no_grad():
            for xb, yb in data_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = self(xb)
                loss = loss_fn(out.squeeze(), yb.float()) # double-check dimensions / should not have to squeeze - test if yb.FLOAT() is necessary
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)

        return avg_loss

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}".format(epoch, result['val_loss']))