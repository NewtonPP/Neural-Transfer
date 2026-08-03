import argparse
import torch 
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image
import torchvision

from model import Decoder, Encoder
from utils import *

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str, default='/Users/prashidha/Desktop/Codes/NeuralTransfer/content_data',
                        help='Location of content dataset')
    parser.add_argument('--style_dir', type=str, default='/Users/prashidha/Desktop/Codes/NeuralTransfer/style_data',
                        help='Location of style dataset')
    parser.add_argument('--vgg', type=str, default='vgg_normalised.pth',
                        help='Location of pre-trained VGG')
    parser.add_argument('--experiment', type=str, default='experiment1',
                        help='Name of experiment')
    parser.add_argument('--final_size', type=int, default=256,
                        help='Size of final image')
    parser.add_argument('--content_size', type=int, default=512,
                        help='Size of content image')
    parser.add_argument('--style_size', type=int, default=512,
                        help='Size of style image')
    parser.add_argument('--crop', action='store_true', default=True,
                        help='Crop image')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5,
                        help='Learning rate decay')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Content weight')
    parser.add_argument('--style_weight', type=float, default=5,
                        help='Style weight')
    
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Log interval')
    
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save interval')
    
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')
    
    parser.add_argument('--decoder_path', type=str, default=None,
                        help='Path to decoder checkpoint')
    
    parser.add_argument('--optimizer_path', type=str, default=None,
                        help='Path to optimizer checkpoint')      


    return parser.parse_args()


def main():
    args = parse_arguments()
    content_path = args.content_dir
    style_path = args.style_dir
    vgg_weights = args.vgg

    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(size=args.style_size, crop=args.crop, final_size=args.final_size)

    content_data = ImageFolderDataset(root=content_path, transform=content_transform)
    style_data = ImageFolderDataset(root=style_path, transform=style_transform)

    content_dataloader = DataLoader(content_data, args.batch_size, True, pin_memory=True, drop_last=True)
    style_dataloader = DataLoader(style_data, args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

    print(f"Number of batches in content dataset: {content_dataloader.__len__()}")
    print(f"Number of batches in style dataset: {style_dataloader.__len__()}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    save_dir = Path('experiment') / args.experiment 
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')

    encoder = Encoder(vgg_weights).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda epoch:  1.0 / (1.0 + args.lr_decay * epoch)
    )

    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path))
        
    print('Training...')

    mse_loss = torch.nn.MSELoss()

    encoder.eval()

    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_dataloader, style_dataloader), total=min(len(content_dataloader), len(style_dataloader)))
        num_batches = min(len(content_dataloader), len(style_dataloader))
        running_loss = 0
        running_closs = 0
        running_sloss = 0

        for content_batch, style_batch in progress_bar:

            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            # To do ADAIN 
            t = AdaIN(content_features=c_feats[-1], style_features=s_feats[-1])

            g = decoder(t)

            g_feats = encoder(g)

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight
            loss_s = 0

            for g_f, s_f in zip(g_feats, s_feats):
                g_f_mean, g_f_std = calc_mean_std(g_f)
                s_f_mean, s_f_std = calc_mean_std(s_f)

                loss_s += mse_loss(g_f_mean, s_f_mean) + mse_loss(g_f_std, s_f_std)

            loss_s = loss_s * args.style_weight

            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Loss:{loss.item():4f}, Content Loss: {loss_c.item():4f}, Style Loss: {loss_s.item():4f}')

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()


        scheduler.step()


        running_loss /= num_batches
        running_closs /= num_batches
        running_sloss /= num_batches


        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter {epoch+1}: Loss:{running_loss:4f}, Content Loss: {running_closs:4f}, Style Loss: {running_sloss:4f}')

        if (epoch+1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)


if __name__ == "__main__":
    main() 




    