import argparse
import os
import torch
from torch import nn
from torch.autograd import Variable
from diffNpatch import diff_model_state, patch_model_state
from models import densenet, channel_selection, resnet
from torchvision import datasets, transforms
import numpy as np

def print_banner(message):
    print(f"\n{'='*50}\n{message}\n{'='*50}\n")

def print_table(headers, rows):
    max_lengths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            max_lengths[i] = max(max_lengths[i], len(str(cell)))
    
    def format_row(row):
        return " | ".join(f"{str(cell):<{max_lengths[i]}}" for i, cell in enumerate(row))
    
    table = [format_row(headers)]
    table.append("-+-".join("-" * length for length in max_lengths))
    table.extend(format_row(row) for row in rows)
    
    print("\n" + "\n".join(table) + "\n")

def test(model, args):
    """Test the model and print accuracy."""
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10('./data.cifar10', train=False, download=True, transform=transform)
    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100('./data.cifar100', train=False, download=True, transform=transform)
    else:
        raise ValueError("Invalid dataset. Choose 'cifar10' or 'cifar100'.")

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model.eval()
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.1f}%)\n')
    return accuracy

def split(args):
    """Split the model and save the keyed version and key."""
    model = resnet(depth=args.depth, dataset=args.dataset) if args.arch == "resnet" else densenet(depth=args.depth, dataset=args.dataset)

    if args.cuda:
        model.cuda()

    print_banner(f"Loading checkpoint '{args.model}'")
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded checkpoint '{args.model}' (epoch {checkpoint['epoch']}) Prec1: {checkpoint['best_prec1']}")

    acc = test(model, args)
    print(f"Original model accuracy: {acc}%")

    total = sum(m.weight.data.shape[0] for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    bn_weights = torch.zeros(total)
    index = 0

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn_weights[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    sorted_bn_weights, _ = torch.sort(bn_weights)
    threshold_index = int(total * (1 - args.ratio))
    threshold = sorted_bn_weights[threshold_index]

    extracted_channels = 0
    cfg = []
    cfg_mask = []

    details = []

    for layer_index, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.le(threshold).float()

            if args.cuda:
                mask = mask.cuda()

            extracted_channels += mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            if args.config:
                details.append([layer_index, mask.shape[0], int(torch.sum(mask))])
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    if args.config:
        print_table(["Layer Index", "Total Channels", "Remaining Channels"], details)

    print('Pre-processing successful!')

    new_model = densenet(depth=args.depth, dataset=args.dataset, cfg=cfg) if args.arch == "densenet" else resnet(depth=args.depth, dataset=args.dataset, cfg=cfg)

    if args.cuda:
        new_model.cuda()

    num_parameters = sum(p.numel() for p in new_model.parameters())
    savepath = os.path.join(args.save, "extraction.txt")
    with open(savepath, "w") as fp:
        fp.write(f"Configuration: \n{cfg}\n")
        fp.write(f"Number of parameters: \n{num_parameters}\n")
        fp.write(f"Test accuracy: \n{acc}\n")

    old_modules = list(model.modules())
    new_modules = list(new_model.modules())

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    first_conv = True

    for layer_id, (m0, m1) in enumerate(zip(old_modules, new_modules)):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            idx1 = np.resize(idx1, (1,)) if idx1.size == 1 else idx1

            if isinstance(old_modules[layer_id + 1], channel_selection):
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                end_mask = cfg_mask[layer_id_in_cfg] if layer_id_in_cfg < len(cfg_mask) else end_mask
                continue

        elif isinstance(m0, nn.Conv2d):
            if first_conv:
                m1.weight.data = m0.weight.data.clone()
                first_conv = False
                continue
            if isinstance(old_modules[layer_id - 1], channel_selection):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                idx0 = np.resize(idx0, (1,)) if idx0.size == 1 else idx0
                idx1 = np.resize(idx1, (1,)) if idx1.size == 1 else idx1
                m1.weight.data = m0.weight.data[:, idx0.tolist(), :, :].clone()
                continue

        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx0 = np.resize(idx0, (1,)) if idx0.size == 1 else idx0
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    acc = test(new_model, args)
    print(f"Keyed model accuracy: {acc}%")

    keyed_model_path = os.path.join(args.save, 'keyed_model.pth.tar')
    key_path = os.path.join(args.save, 'key.pth.tar')

    diff_state_dict = diff_model_state(new_model, model)
    torch.save({'cfg': cfg, 'state_dict': new_model.state_dict()}, keyed_model_path)
    torch.save({"diff_state_dict": diff_state_dict}, key_path)

    print(f"Keyed model saved to: {keyed_model_path}")
    print(f"Key saved to: {key_path}")

def recover(args):
    """Recover the model using the keyed model and the key."""
    if not os.path.isfile(args.model) or not args.key:
        raise ValueError("Invalid model path or key path.")

    print_banner(f"Loading keyed model '{args.model}'")
    ckpt = torch.load(args.model)
    print(f"Loading key '{args.key}'")
    key = torch.load(args.key)

    state_dict = ckpt["state_dict"]
    diff_state_dict = key["diff_state_dict"]
    cfg = ckpt["cfg"]

    new_model = resnet(depth=args.depth, dataset=args.dataset, cfg=cfg) if args.arch == "resnet" else densenet(depth=args.depth, dataset=args.dataset, cfg=cfg)
    old_model = resnet(depth=args.depth, dataset=args.dataset) if args.arch == "resnet" else densenet(depth=args.depth, dataset=args.dataset)

    new_model.load_state_dict(state_dict)
    recovered_model = patch_model_state(new_model, old_model, diff_state_dict)

    test(recovered_model, args)
    torch.save({"state_dict": recovered_model.state_dict()}, os.path.join(args.save, "recovered_model.pth.tar"))

def main(args):
    """Main function to execute split or recover commands."""
    if args.command == "split":
        split(args)
    elif args.command == "recover":
        recover(args)
    else:
        raise ValueError("Invalid command. Use 'split' or 'recover'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyed model")
    parser.add_argument("--use_cuda", action="store_true", default=False, help="Use CUDA training")
    
    sub_parsers = parser.add_subparsers(dest="command", help="Commands: split or recover")
    
    # Split command arguments
    split_cmd_parser = sub_parsers.add_parser("split", help="Split model")
    split_cmd_parser.add_argument("--model", type=str, required=True, metavar="PATH", help="Path to the model")
    split_cmd_parser.add_argument("--arch", type=str, required=True, choices=["resnet", "densenet"], default="densenet", help="Architecture of model")
    split_cmd_parser.add_argument("--ratio", type=float, default=0.01, help="Scale sparse rate (default: 0.01)")
    split_cmd_parser.add_argument("--save", type=str, metavar="PATH", help="Path to save keyed model")
    split_cmd_parser.add_argument("--dataset", type=str, default="cifar10", help="Training dataset (default: cifar10)")
    split_cmd_parser.add_argument("--test-batch-size", type=int, default=256, metavar="N", help="Input batch size for testing (default: 256)")
    split_cmd_parser.add_argument("--depth", type=int, default=40, help="Depth of the network")
    split_cmd_parser.add_argument("--config", action="store_true", default=False, help="Print layer configuration details")

    
    # Recover command arguments
    recover_cmd_parser = sub_parsers.add_parser("recover", help="Recover model")
    recover_cmd_parser.add_argument("--model", type=str, required=True, metavar="PATH", help="Path to the keyed model")
    recover_cmd_parser.add_argument("--key", type=str, required=True, metavar="PATH", help="Path to key")
    recover_cmd_parser.add_argument("--arch", type=str, required=True, choices=["resnet", "densenet"], default="densenet", help="Architecture of model")
    recover_cmd_parser.add_argument("--dataset", type=str, default="cifar10", help="Training dataset (default: cifar10)")
    recover_cmd_parser.add_argument("--test-batch-size", type=int, default=256, metavar="N", help="Input batch size for testing (default: 256)")
    recover_cmd_parser.add_argument("--save", type=str, metavar="PATH", help="Path to save recovered model")
    recover_cmd_parser.add_argument("--depth", type=int, default=40, help="Depth of the network")

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    main(args)