import os, argparse
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import random

import models
from dataloader import get_id_data, get_perturbation_data
from utils import compute_discrepancy, compute_metrics, print_perturbation_results

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='mnist | svhn | cifar10')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--datadir', default='./data/', help='path to dataset')
parser.add_argument('--outdir', default='./discrepancy/', help='folder to output discrepancy estimation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for data loader')
parser.add_argument('--num_epochs', type=int, default=206, help='the number of epochs for training Deep-Tube')
parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate of SGD optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor for nesterov momentum of SGD optimizer')
parser.add_argument('--alpha', type=float, default=0.1, help='regularization coefficient balancing distance and density')
parser.add_argument('--lambda_', type=float, default=0.1, help='regularization coefficient')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
args = parser.parse_args()
print(args)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.set_device(args.gpu)


def get_num_classes(dataset):
    if dataset == 'svhn':
        num_classes = 10
    elif dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'mnist':
        num_classes = 10
    
    return num_classes


def get_perturbation_list(dataset):
    if dataset == 'svhn':
        perturbation_list = ['gaussian', 'uniform', 'mnist', 'cifar10', 'imagenet_crop', 'lsun_crop']
    elif dataset == 'cifar10':
        perturbation_list = ['gaussian', 'uniform', 'mnist', 'svhn', 'imagenet_crop', 'lsun_crop']
    elif dataset == 'mnist':
        perturbation_list = ['gaussian', 'uniform', 'svhn', 'cifar10', 'imagenet_crop', 'lsun_crop']
    
    return perturbation_list


def get_model(net_type, num_classes):
    if net_type == 'densenet':
        model = models.DenseNet_Poincare(num_classes=num_classes)
    elif net_type == 'resnet':
        model = models.ResNet_Poincare(num_classes=num_classes)
    
    return model


def get_transform(net_type):
    if net_type == 'densenet':
        in_transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(), 
                            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                            transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
                            ])
        in_transform_test = transforms.Compose([transforms.ToTensor(), 
            transforms.Lambda(lambda x: torch.nn.functional.pad(x, (2, 2, 2, 2), 'constant', 0) if x.size(-1) == 28 else x),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),])

    elif net_type == 'resnet':
        in_transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(), 
                            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])
        in_transform_test = transforms.Compose([transforms.ToTensor(), 
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        
    return in_transform_train, in_transform_test


def main():
    set_random_seed(0)

    num_classes = get_num_classes(args.dataset) 
    model = get_model(args.net_type, num_classes).cuda()
    in_transform_train, in_transform_test = get_transform(args.net_type)
    train_loader, test_id_loader = get_id_data(args.dataset, args.batch_size, in_transform_train, in_transform_test, args.datadir)
    
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-4)

    perturbation_list = get_perturbation_list(args.dataset)
    # set the path to discrepancy estimation
    outdir = os.path.join(args.outdir, args.net_type + '_' + args.dataset)
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)

    for epoch in range(args.num_epochs):
        print('current learning rate:', optimizer.param_groups[-1]['lr'])
        model.train()
        total_loss = 0.0
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            images, labels = images.cuda(), labels.cuda()
            dists1, dists2 = model(images)

            scores1 = - dists1 + model.alphas1
            scores2 = - dists2 + model.alphas2
            label_mask = torch.zeros(labels.size(0), num_classes).cuda().scatter_(1, labels.unsqueeze(dim=1), 1)
            pull_loss1, push_loss1 = torch.mean(torch.sum(torch.mul(label_mask, dists1), dim=1)), ce_loss(scores1, labels)
            pull_loss2, push_loss2 = torch.mean(torch.sum(torch.mul(label_mask, dists2), dim=1)), ce_loss(scores2, labels)

            loss1 = args.lambda_ * pull_loss1 + push_loss1
            loss2 = args.lambda_ * pull_loss2 + push_loss2
            loss = loss1 + args.alpha * loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
   
        scheduler.step()

        if epoch < 5 or (epoch > 10 and epoch % 10 == 5):
            model.eval()
            with torch.no_grad():
                # (1) evaluate ID classification
                correct, total = 0, 0
                for images, labels in test_id_loader:
                    images, labels = images.cuda(), labels.cuda()
                    dists1, _ = model(images)
                    scores1 = - dists1 + model.alphas1
                    _, predicted = torch.max(scores1, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                idacc = 100 * correct / total

                # (2) evaluate perturbation elimination                    
                perturbation_results_list = []
                compute_discrepancy(model, test_id_loader, outdir, True)

                for pert in perturbation_list:
                    test_perturbation_loader = get_perturbation_data(pert, args.batch_size, in_transform_test, args.datadir)
                    compute_discrepancy(model, test_perturbation_loader, outdir, False)
                    perturbation_results_list.append(compute_metrics(outdir))
            
            print('== Epoch [{}/{}], Loss {} =='.format(epoch+1, args.num_epochs, total_loss))   
            print('In-dist Accuracy on "{idset:s}" test images : {val:6.5f}\n'.format(idset=args.dataset, val=idacc))
            for pert_idx, pert_results in enumerate(perturbation_results_list):
                print('Perturbation accuracy on "{pertset:s}":'.format(pertset=perturbation_list[pert_idx]))
                print_perturbation_results(pert_results)


if __name__ == '__main__':
    main()
