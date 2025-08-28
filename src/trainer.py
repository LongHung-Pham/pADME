import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import torch.optim as optim


class MultitaskTrainer:
    """
    L1Loss(reduction='none') for custom masking (multitask pretraining)
    L1Loss(reduction='mean') for single-task fine-tuning
    """
    def __init__(self, model, tasks=['y_sol', 'y_logd', 'y_hlm', 'y_mlm', 'y_mdck'], mode = 'pretrain', device='cuda'):
        self.model = model
        self.tasks = tasks
        self.device = device
        self.model.to(device)

        if mode == 'pretrain':
            self.criterion = nn.L1Loss(reduction='none')      # 'none' for custom masking
        elif mode == 'finetune':
            self.criterion = nn.L1Loss()

    def pretrain(self, train_dataset, test_dataset, epochs = 200, lr = 1e-4, task_weights=None):
        """
        Pretrain the model on all tasks simultaneously.

        Args:
            train_dataset: The training dataset
            test_dataset: The test dataset
            batch_size: Batch size for training
            epochs: Number of training epochs
            lr: Learning rate
            task_weights: Optional dictionary of weights for each task's loss
        """
        self.model.set_fine_tuning_mode(None)

        if task_weights is None:
            task_weights = {task: 1.0 for task in self.tasks}

        train_loader = DataLoader(train_dataset, batch_size = 8192, shuffle = True)
        test_loader = DataLoader(test_dataset, batch_size = 4096)

        optimizer = optim.AdamW(self.model.parameters(), lr = lr)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, steps_per_epoch = len(train_loader), epochs = epochs)

        best_loss = float('inf')

        for epoch in range(epochs):

            self.model.train()
            train_losses = {task: 0.0 for task in self.tasks}
            train_counts = {task: 0 for task in self.tasks}
            total_loss = 0.0

            for batch_idx, data in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(data)

                # Calculate loss for each task and weighted sum
                batch_loss = 0.0
                for task in self.tasks:
                    target = getattr(data, task)
                    mask = ~torch.isnan(target)         # non-NaN values
                    if not mask.any():
                        continue

                    task_losses = self.criterion(outputs[task].squeeze()[mask], target[mask])
                    # Reduce to scalar and apply task weight
                    task_loss = task_losses.mean() * task_weights[task]
                    batch_loss += task_loss

                    train_losses[task] += task_loss.item() * mask.sum().item()
                    train_counts[task] += mask.sum().item()

                if batch_loss > 0:                      # Only backward if we have a valid loss
                    total_loss += batch_loss.item()
                    batch_loss.backward()
                    optimizer.step()
                    scheduler.step()

            # Calculate average losses
            for task in self.tasks:
                if train_counts[task] > 0:
                    train_losses[task] /= train_counts[task]
                else:
                    train_losses[task] = float('nan')

            # Evaluation
            self.model.eval()
            test_losses = {task: 0.0 for task in self.tasks}
            test_counts = {task: 0 for task in self.tasks}
            test_total_loss = 0.0

            with torch.no_grad():
                for data in test_loader:
                    data = data.to(self.device)
                    outputs = self.model(data)

                    batch_loss = 0.0
                    for task in self.tasks:
                        target = getattr(data, task)
                        mask = ~torch.isnan(target)
                        if not mask.any():
                            continue

                        task_losses = self.criterion(outputs[task].squeeze()[mask], target[mask])

                        task_loss = task_losses.mean() * task_weights[task]
                        batch_loss += task_loss

                        test_losses[task] += task_loss.item() * mask.sum().item()
                        test_counts[task] += mask.sum().item()

                    if batch_loss > 0:
                        test_total_loss += batch_loss.item()

            for task in self.tasks:
                if test_counts[task] > 0:
                    test_losses[task] /= test_counts[task]
                else:
                    test_losses[task] = float('nan')


            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {total_loss:.4f} | ' +
                  ' | '.join([f'{task}: {loss:.4f}' for task, loss in train_losses.items()]))
            print(f'  Test Loss: {test_total_loss:.4f} | ' +
                  ' | '.join([f'{task}: {loss:.4f}' for task, loss in test_losses.items()]))

            # Save best model
            if test_total_loss < best_loss:
                best_loss = test_total_loss
                torch.save(self.model.state_dict(), 'best_pretrained_Novartis_full_data.pt')
                print('  Model saved.')

        return self.model


    def finetune(self, task_name, train_dataset, test_dataset, epochs = 50, lr = 1e-4):
        """
        Fine-tune the model for a specific task.

        Args:
            task_name: The name of the task to fine-tune on
            train_dataset: The training dataset
            test_dataset: The test dataset
            batch_size: Batch size for training (smaller for fine-tuning)
            epochs: Number of training epochs
            lr: Learning rate (lower for fine-tuning)
        """
        self.model.set_fine_tuning_mode(task_name)

        train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size = 16)

        # Optimizer - only optimize the task-specific parameters
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, steps_per_epoch = len(train_loader), epochs = epochs)

        best_loss = float('inf')
        smallest_difference = float('inf')

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()

                output = self.model(data)                                           # Now returns only the task-specific output
                loss = self.criterion(output.squeeze(), getattr(data, 'y'))         # task_name

                train_loss += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Calculate average loss
            train_loss /= len(train_dataset)

            # Evaluation
            self.model.eval()
            test_loss = 0.0

            with torch.no_grad():
                for data in test_loader:
                    data = data.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output.squeeze(), getattr(data, 'y'))    # task_name
                    test_loss += loss.item() * data.num_graphs

            test_loss /= len(test_dataset)

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Test Loss: {test_loss:.4f}')

            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(self.model.state_dict(), f'best_finetuned_model_{task_name}_Novartis.pt')
                print('  Model saved.')

        return self.model

    def cross_val(self, task_name, train_dataset, test_dataset, epochs = 50, lr = 1e-4):
        self.model.set_fine_tuning_mode(task_name)

        train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size = 16)

        # Optimizer - only optimize the task-specific parameters
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, steps_per_epoch = len(train_loader), epochs = epochs)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()

                output = self.model(data)                                           # Now returns only the task-specific output
                loss = self.criterion(output.squeeze(), getattr(data, 'y'))         # task_name

                train_loss += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Calculate average loss
            train_loss /= len(train_dataset)

        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                loss = self.criterion(output.squeeze(), getattr(data, 'y'))         # task_name
                test_loss += loss.item() * data.num_graphs
        test_loss /= len(test_dataset)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Fold MAE: {test_loss:.4f}')

        return test_loss