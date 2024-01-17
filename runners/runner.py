import numpy as np
import glob
from tqdm import tqdm
from losses.dsm import anneal_dsm_score_estimation
from collections import deque
import torch.nn.functional as F
import logging
import torch
import os
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from models.net import RefineNet
from datasets import get_dataset, data_transform, inverse_data_transform
from losses import get_optimizer
from models import anneal_Langevin_dynamics, anneal_Langevin_dynamics_original
from models import get_sigmas
from models.ema import EMAHelper



def get_model(config):
    if config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA' or config.data.dataset == 'FashionMNIST' or config.data.dataset == 'MNIST' or config.data.dataset == 'CUB':
        return RefineNet(config).to(config.device)

# +
class Runner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)
        
        N = self.config.training.queue_size
        self.p_sample_buff = deque(maxlen=N)
        self.sample_buff = deque(maxlen=N)
        self.label_buff = deque(maxlen=N)
        self.sigma_buff = deque(maxlen=N)


    def train(self):
        dataset, test_dataset = get_dataset(self.args, self.config)

        # show one of the ground truth image samples
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=36, shuffle=True)
        images, _ = next(iter(train_loader))
        grid = make_grid(images, nrow=6)
        save_image(grid, os.path.join(self.args.log_sample_path, 'ground_truth_sample.png'))


        dataloader = DataLoader(dataset, batch_size=self.config.training.k, shuffle=True,
                                num_workers=self.config.data.num_workers, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=True)
        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_logger = self.config.tb_logger

        score = get_model(self.config)

        score = torch.nn.DataParallel(score)
        optimizer = get_optimizer(self.config, score.parameters())

        start_epoch = 0
        step = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        sigmas = get_sigmas(self.config)
        N = 1000

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                # shape of y : (num_images, 40)
                # attribute selection
                #if self.config.sampling.private_attribute == 'gender':
                #    y = y[:, 20:21]
                #elif self.config.sampling.private_attribute == 'smile':
                #    y = y[:, 31:32]
                #else:
                #    y = y[:, 2:3]
                # target = torch.eye(2)[y].squeeze().cuda()
                # print(target.shape)
                score.train()
                step += 1

                X = X.to(self.config.device)
                X = data_transform(self.config, X)
                # import ipdb; ipdb.set_trace()

                # init buff
                self.p_sample_buff = deque(maxlen=N)
                self.sample_buff = deque(maxlen=N)
                self.label_buff = deque(maxlen=N)
                self.sigma_buff = deque(maxlen=N)
                kwargs = {'p_sample_buff': self.p_sample_buff, 'sample_buff': self.sample_buff, 
                        'label_buff':self.label_buff, 'sigma_buff':self.sigma_buff, 'config': self.config}

                losses = anneal_dsm_score_estimation(score, X, sigmas, None,
                                                   self.config.training.anneal_power,
                                                   hook=None, **kwargs)
                
                if losses == 'pass':
                    print(len(self.sample_buff))
                    continue
                    
                loss = losses.mean(dim=0)
                logging.info("step: {}, loss: {}".format(step, losses.mean().item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(score)

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    if self.config.model.ema:
                        test_score = ema_helper.ema_copy(score)
                    else:
                        test_score = score

                    test_score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = data_transform(self.config, test_X)

                    # with torch.no_grad():
                    #     # test_dsm_loss = anneal_dsm_score_estimation(test_score, test_X, sigmas, None,
                    #     #                                             self.config.training.anneal_power,
                    #     #                                             hook=test_hook)
                    #     test_dsm_losses = anneal_dsm_score_estimation(test_score, test_X, sigmas, None,
                    #                                                 self.config.training.anneal_power)
                    #     # tb_logger.add_scalar('test_loss', test_dsm_loss, global_step=step)
                    #     # logging.info("step: {}, test_loss: {}".format(step, test_dsm_loss.item()))
                    #     logging.info("step: {}, test_loss: {}".format(step, test_dsm_losses.mean().item()))

                    #     del test_score

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pth'))

                    if self.config.training.snapshot_sampling:
                        if self.config.model.ema:
                            test_score = ema_helper.ema_copy(score)
                        else:
                            test_score = score

                        test_score.eval()
                        init_samples = torch.rand(self.config.training.sample_size, 
                                                  self.config.data.channels,
                                                  self.config.data.image_size, self.config.data.image_size,
                                                  device=self.config.device, requires_grad=True)
                        init_samples = data_transform(self.config, init_samples)
                        indices = torch.randperm(len(y))[:self.config.training.sample_size]
                        shuffle_labels = y[indices]

                        all_samples = anneal_Langevin_dynamics_original(init_samples,
                                                               test_score, sigmas.cpu().numpy(),
                                                               self.config.sampling.n_steps_each,
                                                               self.config.sampling.step_lr,
                                                               final_only=True, verbose=True,
                                                               denoise=self.config.sampling.denoise)

                        sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                      self.config.data.image_size,
                                                      self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, 3)
                        save_image(image_grid,
                                   os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)))
                        torch.save(sample, os.path.join(self.args.log_sample_path, 'samples_{}.pth'.format(step)))

                        del test_score
                        del all_samples


    def fast_fid(self):
        ### Test the fids of ensembled checkpoints.
        ### Shouldn't be used for models with ema
        if self.config.fast_fid.ensemble:
            if self.config.model.ema:
                raise RuntimeError("Cannot apply ensembling to models with EMA.")
            self.fast_ensemble_fid()
            return

        from evaluation.fid_score import get_fid, get_fid_stats_path
        import pickle
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        fids = {}
        self.args.log_path = '/data/local/xinxi/Project/DPgan_model/logs/exp_cub/logs/CUB'
        for ckpt in tqdm(range(self.config.fast_fid.begin_ckpt, self.config.fast_fid.end_ckpt + 1, 5000),
                              desc="processing ckpt"):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(score)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(score)
            else:
                score.load_state_dict(states[0])

            score.eval()

            num_iters = self.config.fast_fid.num_samples // self.config.fast_fid.batch_size
            # output_path = os.path.join(self.args.image_folder, 'ckpt_{}'.format(ckpt))
            output_path = '/data/local/xinxi/Project/DPgan_model/logs/exp_cub/fid_samples'
            os.makedirs(output_path, exist_ok=True)

            """dataset, _ = get_dataset(self.args, self.config)
            label_loader = torch.utils.data.DataLoader(dataset, 
                                                       batch_size=self.config.fast_fid.batch_size*2, 
                                                       shuffle=True)
            _, labels = next(iter(label_loader))
            if self.config.sampling.private_attribute == 'gender':
                    labels = labels[:, 20:21]
            elif self.config.sampling.private_attribute == 'smile':
                    labels = labels[:, 31:32]
            else:
                    labels = labels[:, 2:3]
            indices = torch.randperm(len(labels))[:self.config.fast_fid.batch_size]
            shuffle_labels = labels[indices]"""

            for i in range(num_iters):
                init_samples = torch.rand(self.config.fast_fid.batch_size, self.config.data.channels,
                                          self.config.data.image_size, self.config.data.image_size,
                                          device=self.config.device, requires_grad=True)
                init_samples = data_transform(self.config, init_samples)

                """all_samples = anneal_Langevin_dynamics(init_samples, shuffle_labels, 
                                                       self.config.sampling.private_attribute,
                                                       score, sigmas,
                                                       self.config.fast_fid.n_steps_each,
                                                       self.config.fast_fid.step_lr,
                                                       verbose=self.config.fast_fid.verbose,
                                                       denoise=self.config.sampling.denoise)"""
                
                all_samples = anneal_Langevin_dynamics_original(init_samples,
                                                               score, sigmas,
                                                               self.config.sampling.n_steps_each,
                                                               self.config.sampling.step_lr,
                                                               final_only=True, verbose=True,
                                                               denoise=self.config.sampling.denoise)

                final_samples = all_samples[-1]
                for id, sample in enumerate(final_samples):
                    sample = sample.view(self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(id)))

            stat_path = get_fid_stats_path(self.args, self.config, download=True)
            fid = get_fid(stat_path, output_path)
            fids[ckpt] = fid
            print("ckpt: {}, fid: {}".format(ckpt, fid))

        with open(os.path.join(self.args.image_folder, 'fids.pickle'), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def is_score(self):
        import pathlib
        from evaluation.fid_score import get_activations
        from evaluation.inception import InceptionV3
        import pickle

        # inception score test
        base_path = '/data/local/qipan/exp_celeba/dpdm_samples'
        checkpoint_paths = os.listdir(base_path)

        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])
        if torch.cuda.is_available():
            model.cuda()

        inception_score_dict = {}
        for path in checkpoint_paths:
            print(path)
            if path.endswith('.pickle'):
                continue
            ckpt_path = pathlib.Path(os.path.join(base_path, path))
            images_path = list(ckpt_path.glob('*.jpg')) + list(ckpt_path.glob('*.png'))

            activations = get_activations(images_path, model, 50, dims, True)
            activations = torch.from_numpy(activations)
            # Calculate marginal distribution
            p_yx = F.softmax(activations, dim=1).mean(dim=0)
            
             # Calculate scores for each image
            scores = []
            splits = 10

            for i in range(splits):
                # Randomly sample from the activations
                part = activations[i * (activations.shape[0] // splits):(i + 1) * (activations.shape[0] // splits)]
                # Calculate the conditional distribution
                p_yx_i = F.softmax(part, dim=1).mean(dim=0)
                # Calculate KL divergence
                kl_div = (p_yx_i * (p_yx_i.log() - p_yx.log())).sum()
                scores.append(torch.exp(kl_div).item())

            # Calculate the Inception Score
            inception_score = np.mean(scores)
            inception_score_dict[ckpt_path] = inception_score
            print("ckpt: {}, inception_score: {}".format(ckpt_path, inception_score))

        with open(os.path.join(base_path, 'is_score.pickle'), 'wb') as handle:
            pickle.dump(inception_score_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def privacy_eval(self):
        import pathlib
        from evaluation.fid_score import get_activations
        from evaluation.inception import InceptionV3
        import pickle
        from torchvision import transforms
        from PIL import Image
        from evaluation.privacy_eval import is_private 
        

        # privacy eval test
        dataset_path = os.path.join(self.args.exp, 'datasets', 'celeba/celeba/img_align_celeba')
        gtimages_path = os.listdir(dataset_path)
        base_path = '/data/local/qipan/exp_celeba/dpdm_samples/samples'

        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])
        if torch.cuda.is_available():
            model.cuda()

        private_score_dict = {}
        images_path = list(pathlib.Path(base_path).glob('*.jpg')) + list(pathlib.Path(base_path).glob('*.png'))

        # private score init
        private_score = 0
        total_sample = 0
        for id, image in enumerate(images_path):
            # we just sample 40 images for each checkpoint
            if id >= 40:
                break

            image_list = [image]
            print(image_list)
            activations_gen = get_activations(image_list, model, 1, dims, True)
            activations_gen = np.squeeze(activations_gen, axis=0)
            # activation shape: (2048, )
            min_dist = np.full_like(activations_gen, np.inf)
            nearest_img_path = None

            for gtimage_path in tqdm(gtimages_path):
                gt_image_list = [os.path.join(dataset_path, gtimage_path)]
                activations_real = get_activations(gt_image_list, model, 1, dims, True)
                activations_real = np.squeeze(activations_real, axis=0)
                # print(nearest_img_path)
                # Calculate L2 distance
                dist = np.linalg.norm(activations_gen - activations_real)
                if np.sum(dist) < np.sum(min_dist):
                    min_dist = dist
                    nearest_img_path = gt_image_list[0]

            if nearest_img_path is not None:
                concat_images_path = [nearest_img_path, image]
                concat_images = [Image.open(path) for path in concat_images_path]
                # reshape ground truth image size
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                ])
                concat_images = [transform(image) for image in concat_images]
                image_grid = make_grid(concat_images, 1)
                save_path = os.path.join(self.args.exp, 'dpdm_samples', 
                                        'nearest_image', self.config.sampling.private_attribute)
                os.makedirs(save_path, exist_ok=True)
                save_image(image_grid, os.path.join(save_path, '{}.png'.format(id)))

                # calculate private score
                total_sample += 1
                print(concat_images[0].unsqueeze(0).shape)
                print(concat_images[1].unsqueeze(0).shape)
                private_score += is_private(concat_images[0].unsqueeze(0).cuda(), 
                                            concat_images[1].unsqueeze(0).cuda(),
                                            self.config.sampling.private_attribute)
        
        print("ckpt: {}, private_score: {}".format("dpdm", private_score/total_sample))
        private_score_dict["dpdm"] = private_score/total_sample
        
        with open(os.path.join(save_path, '{}.pickle'.format(self.config.sampling.private_attribute)), 'wb') as handle:
            pickle.dump(private_score_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


            
                
            


            
            

    
