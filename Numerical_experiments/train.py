import torch
import os

def train(model, device, train_loader, test_loader, criterion, optimizer, epoch, save_path, save_iterations, metrics):
    """
    Train the model for one epoch, save model checkpoints at specified iterations, 
    and update performance metrics.
    """
    model.train()

    # --- 1. Identify the specific weight layer to track ---
    target_param = None
    
    if hasattr(model, 'classifier'):
        # Case for SimpleCNN:
        # The user requested to save the LAST fully connected layer (Hidden -> Output).
        # Structure of model.classifier:
        # [0] Flatten -> [1] Linear(Feature, Hidden) -> [2] ReLU -> [3] Linear(Hidden, Output)
        target_param = model.classifier[3].weight
    else:
        # Case for FCN (MLP):
        # Maintain original behavior: Save the MIDDLE layer (2nd Linear layer).
        # model.parameters() list order: [L1_w, L1_b, L2_w, L2_b, L3_w, L3_b]
        # Index [2] corresponds to the weight of the second layer.
        target_param = list(model.parameters())[2]

    # --- Internal Helper: Save metrics logic to avoid code duplication ---
    def save_metrics_at_step(metrics_dict):
        # Compute loss and accuracy for train and test sets
        train_loss, train_accuracy = compute_metrics(model, train_loader, device, criterion)
        test_loss, test_accuracy = compute_metrics(model, test_loader, device, criterion)
        
        # Extract weights and convert to numpy
        w_data = target_param.detach().cpu().numpy()
        
        # Dynamic reshape: Flatten into a column vector (N, 1)
        # This makes it compatible with both FCN (2500 elements) and SimpleCNN (5120 elements)
        w_flat = w_data.reshape(-1, 1) 
        
        # Get indices of wrong predictions
        wrong_indices = get_wrong_predictions(model, test_loader, device)

        # Update dictionary
        metrics_dict['train_loss'].append(train_loss)
        metrics_dict['test_loss'].append(test_loss)
        metrics_dict['train_accuracy'].append(train_accuracy)
        metrics_dict['test_accuracy'].append(test_accuracy)
        metrics_dict['weight_all'].append(w_flat) 
        metrics_dict['wrong_indices'].append(wrong_indices)
        return metrics_dict

    # --- 2. Collect performance metrics at iteration 0 (Before training starts) ---
    if epoch == 0:
        checkpoint_path = os.path.join(save_path, "iteration_0.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}") 
        metrics = save_metrics_at_step(metrics)

    # --- 3. Training Loop ---
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Optimization step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Calculate current global iteration (starting from 1)
        current_iteration = int(epoch * len(train_loader) + batch_idx + 1)

        # Print log every 10 batches
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Save Checkpoint and Metrics at specified iterations
        if current_iteration in save_iterations:
            # Save model checkpoint
            checkpoint_path = os.path.join(save_path, f"iteration_{current_iteration}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")    

            # Update metrics
            metrics = save_metrics_at_step(metrics)

    return metrics    

def test(model, device, test_loader, criterion):
    """Test model performance"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Accumulate loss for batches
            pred = output.argmax(dim=1, keepdim=True)  # Get predicted labels with max probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

def compute_metrics(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    model.train()
    return total_loss / total, correct / total


def get_wrong_predictions(model, data_loader, device):
    model.eval()
    wrong_indices = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            wrong_batch_indices = (predicted != target).nonzero(as_tuple=False).reshape(-1).tolist()
            # Calculate global indices since each batch's indices start from 0
            global_indices = [idx + batch_idx * data_loader.batch_size for idx in wrong_batch_indices]
            wrong_indices.extend(global_indices)
    model.train()
    return wrong_indices

