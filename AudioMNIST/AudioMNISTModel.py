import torch
import torch.nn as nn
import torch.nn.init as init


class ConvBlock(nn.Module):
    def __init__(self, activation, in_channels, out_channels, kernel_size=3, pool_size=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.batch = nn.BatchNorm1d(out_channels)
        self.activation = activation
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

class AudioMnistModel(nn.Module):
    def __init__(self, activation, hidden_size, in_channels, output_size, conv_channels=[32,64]):
        super(AudioMnistModel, self).__init__()
        self.conv_blocks = nn.ModuleList()

        for out_channels in conv_channels:
            self.conv_blocks.append(ConvBlock(activation, in_channels, out_channels))
            in_channels = out_channels

        self.adaptive_pool = nn.AdaptiveAvgPool1d(25)
        self.flattened_size = self._calculate_flattened_size()

        layers = []
        prev_size = self.flattened_size
        for size in hidden_size:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(activation)
            prev_size = size
        self.fc_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size[-1], output_size)

        self._initialize_weights()

    def _calculate_flattened_size(self): # mock MNIST model after convolutions
        x = torch.zeros(1, 40, 100)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.adaptive_pool(x)
        return x.view(1, -1).size(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.adaptive_pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc_layers(x)
        x = self.output_layer(x)
        return x

    def training_step(self, batch, loss_func):
        images, labels = batch
        out = self(images)
        loss = loss_func(out, labels)
        return loss

    def train_epoch(self, train_loader, optim, loss_func, device):
        self.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optim.zero_grad()
            out = self(images)
            loss = self.training_step((images, labels), loss_func)
            loss.backward()
            optim.step()

            train_loss += loss.item()

            _, preds = torch.max(out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = train_loss / len(train_loader)
        accuracy_percentage = (correct / total) * 100  # in percentage

        return avg_loss, accuracy_percentage

    def validation_step(self, batch, loss_func):
        images, labels = batch
        device = next(self.parameters()).device
        images, labels = images.to(device), labels.to(device)

        # Compute forward pass only once
        out = self(images)
        loss = loss_func(out, labels)
        # Compute predictions only once from the same output
        _, preds = torch.max(out, dim=1)
        acc = (preds == labels).sum().item() / labels.size(0)
        # Return results as tensors for consistency with validation_epoch_end
        return {'val_loss': torch.tensor(loss.item()), 'val_acc': torch.tensor(acc)}

    def validate_epoch(self, val_loader, loss_func, device):
        self.eval()
        val_results = []
        with torch.no_grad():
            for batch in val_loader:
                result = self.validation_step(batch, loss_func)
                val_results.append(result)
        overall = self.validation_epoch_end(val_results)
        return overall, overall['val_acc']

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
