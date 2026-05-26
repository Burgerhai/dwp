import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_DWP(net, train_loader, criterion, optimizer, device, noise_std, init_num=10, model = 'cnn'):
    net.train()
    correct = 0
    clean_losssum = 0.0
    noise_loss_mean_sum = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        if model == 'cnn':
            outputs = net(inputs)
        else:
            with torch.backends.cudnn.flags(enabled=False):
                outputs = net(inputs)
        clean_loss = criterion(outputs, labels)

        noise_loss = []
        noise_outputs = []
        for i in range(init_num):
            noise_output = net.forward_with_noise(inputs, noise_std[0])
            noise_loss.append(criterion(noise_output, labels))
            noise_outputs.append(noise_output)
        
        noise_outputs_tensor = torch.stack(noise_outputs)  # [init_num, batch_size, num_classes]
        
        scores = max_diversity_selection(noise_outputs_tensor)
        
        noise_loss_mean = 0
        for i in range(len(scores)):
            noise_loss_mean += noise_loss[i] * scores[i]

        loss = clean_loss + noise_loss_mean

        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        clean_losssum += clean_loss.item() * labels.size(0)
        noise_loss_mean_sum += noise_loss_mean.item() * labels.size(0)
    
    train_acc = correct / len(train_loader.dataset)
    clean_loss = clean_losssum / len(train_loader.dataset)
    noise_loss_mean = noise_loss_mean_sum / len(train_loader.dataset)

    return train_acc, clean_loss, noise_loss_mean

def max_diversity_selection(noise_outputs):
    n, batch_size, num_classes = noise_outputs.shape
    all_similarity_scores = []
    
    for batch_idx in range(batch_size):
        sample_outputs = noise_outputs[:, batch_idx, :]  # [n, num_classes]
        similarity_matrix = compute_cosine_similarity_tensor(sample_outputs)  # [n, n]
        batch_similarity_scores = []
        for i in range(n):
            other_similarities = torch.cat([similarity_matrix[i, :i], similarity_matrix[i, i+1:]])
            batch_similarity_scores.append(torch.mean(other_similarities).item())
        
        all_similarity_scores.append(batch_similarity_scores)
    all_similarity_scores_tensor = torch.tensor(all_similarity_scores)  # [batch_size, n]
    avg_similarity_scores = torch.mean(all_similarity_scores_tensor, dim=0)  # [n]
    min_score = torch.min(avg_similarity_scores)
    max_score = torch.max(avg_similarity_scores)
    
    if max_score - min_score > 1e-8: 
        normalized_scores = (avg_similarity_scores - min_score) / (max_score - min_score)
    else:
        normalized_scores = torch.ones_like(avg_similarity_scores)
    
    return normalized_scores.tolist()

def compute_cosine_similarity_tensor(tensor):
    normalized_tensor = F.normalize(tensor, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_tensor, normalized_tensor.t())
    
    return similarity_matrix
