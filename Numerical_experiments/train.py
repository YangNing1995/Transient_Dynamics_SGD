import torch
import os

def train(model, device, train_loader, test_loader, criterion, optimizer, epoch, save_path, save_iterations, metrics):
    """Train the model for one epoch, save model checkpoints at specified iterations, and update performance metrics"""
    model.train()

    # Collect performance metrics at iteration 0
    if epoch == 0:
        checkpoint_path = os.path.join(save_path, "iteration_0.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}") 

        train_loss, train_accuracy = compute_metrics(model, train_loader, device, criterion)
        test_loss, test_accuracy = compute_metrics(model, test_loader, device, criterion)
        weight_list = list(model.parameters())[2].detach()  
        wrong_indices = get_wrong_predictions(model, test_loader, device)  # Wrong prediction indices

        metrics['train_loss'].append(train_loss)
        metrics['test_loss'].append(test_loss)
        metrics['train_accuracy'].append(train_accuracy)
        metrics['test_accuracy'].append(test_accuracy)
        metrics['weight_all'].append(weight_list.cpu().view(2500, -1).numpy())  # Save weights
        metrics['wrong_indices'].append(wrong_indices)  # Save indices of wrong predictions

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        current_iteration = int(epoch * len(train_loader) + batch_idx + 1)

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]Loss: {loss.item():.6f}')

        if current_iteration in save_iterations:
            # Save checkpoint
            checkpoint_path = os.path.join(save_path, f"iteration_{current_iteration}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")    

            # Compute metrics on train and test sets
            train_loss, train_accuracy = compute_metrics(model, train_loader, device, criterion)
            test_loss, test_accuracy = compute_metrics(model, test_loader, device, criterion)
            weight_list = list(model.parameters())[2].detach()  
            wrong_indices = get_wrong_predictions(model, test_loader, device)  # Wrong prediction indices

            # Update performance metrics
            metrics['train_loss'].append(train_loss)
            metrics['test_loss'].append(test_loss)
            metrics['train_accuracy'].append(train_accuracy)
            metrics['test_accuracy'].append(test_accuracy)
            metrics['weight_all'].append(weight_list.cpu().numpy().reshape(2500, -1))  # Save weights
            metrics['wrong_indices'].append(wrong_indices)  # Save indices of wrong predictions

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

def get_wrong_predictions_single_batch(model, data, target):
    wrong_indices = []
    outputs = model(data)
    _, predicted = torch.max(outputs, 1)
    wrong_batch_indices = (predicted != target).nonzero(as_tuple=False).squeeze().cpu().numpy()
    wrong_indices.extend(wrong_batch_indices.tolist())   
    return wrong_indices

def get_wrong_predictions(model, data_loader, device):
    model.eval()
    wrong_indices = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            wrong_batch_indices = (predicted != target).nonzero(as_tuple=False).squeeze().tolist()
            # Calculate global indices since each batch's indices start from 0
            global_indices = [idx + batch_idx * data_loader.batch_size for idx in wrong_batch_indices]
            wrong_indices.extend(global_indices)
    model.train()
    return wrong_indices
