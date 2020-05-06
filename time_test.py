from network import Unet
from dataset import CustomDataset
import torch
from torch.utils.data import DataLoader
import torchvision
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import time

torch.set_printoptions(profile='full')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    print("Default path : " + os.getcwd())
    parser.add_argument("--model_dir", default='model/36epo_383000step.ckpt',
                        help="Directory of pre-trained model, you can download at \n"
                             "https://drive.google.com/drive/folders/1s4M-_SnCPMj_2rsMkSy3pLnLQcgRakAe?usp=sharing")
    parser.add_argument('--dataset', help='Directory of your test_image ""folder""', default='dataset/PASCAL-S/images')
    parser.add_argument('--cuda', help="cuda for cuda, cpu for cpu, default = cuda", default='cpu')
    parser.add_argument('--batch_size', help="batchsize, default = 4", default=1, type=int)
    parser.add_argument('--logdir', help="logdir, log on tensorboard", default=None)
    parser.add_argument('--save_dir', help="save result images as .jpg file. If None -> Not save", default=None)

    args = parser.parse_args()


    print(args)
    print(os.getcwd())
    device = torch.device(args.cuda)
    state_dict = torch.load(args.model_dir, map_location=args.cuda)
    model = Unet().to(device)
    model.load_state_dict(state_dict)
    custom_dataset = CustomDataset(root_dir=args.dataset)
    dataloader = DataLoader(custom_dataset, args.batch_size, shuffle=False)
    # os.makedirs(os.path.join(args.save_dir, 'img'), exist_ok=True)
    # os.makedirs(os.path.join(args.save_dir, 'mask'), exist_ok=True)
    if args.logdir is not None:
        writer = SummaryWriter(args.logdir)
    model.eval()
    t0 = time.time()

    t0 = time.time()
    n_images = 100
    for i, batch in enumerate(tqdm(dataloader)):
        if i == n_images: break
        img = batch.to(device)
        with torch.no_grad():
            pred, loss = model(img)

    t_time = time.time() - t0
    print('Images:', n_images)
    print('Time:', t_time)
    print('FPS:', n_images/t_time)
