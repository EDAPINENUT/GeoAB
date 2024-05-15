from .trainer import *
from evaluation import compute_rmsd

class GenTrainer(Trainer):
    def __init__(self, model, train_loader, valid_loader, save_dir, args):
        super().__init__(model, train_loader, valid_loader, save_dir, args)
    

    def train(self, use_rmsd=False):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        with open(os.path.join(self.save_dir, 'train_config.json'), 'w') as fout:
            json.dump(self.args.__dict__, fout, indent=2)
        self.log_file = open(os.path.join(self.save_dir, "train_log.txt"), 'a+')
            
        self.model.to(device)

        self.model.train()
        for _ in range(self.args.max_epoch):
            for batch in tqdm(self.train_loader):
                batch = self.to_device(batch, device)
                
                loss, loss_list = self.model(batch['X'], batch['S'], batch['L'], batch['offsets'])
                self.optimizer.zero_grad()
                loss.backward()
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()

                if self.global_step % 100 == 1: 
                    print("\033[0;30;46m {} | Epoch: {}, Step: {} | Train Loss: {:.5f}, Loss list:{}\033[0m".format(time.strftime("%Y-%m-%d %H-%M-%S"), self.epoch, self.global_step, loss.item(), ["%0.4f" % i for i in loss_list]))
                    self.log_file.write("{} | Epoch: {}, Step: {} | Train Loss: {:.5f}, Loss list:{}\n".format(time.strftime("%Y-%m-%d %H-%M-%S"), self.epoch, self.global_step, loss.item(), ["%0.4f" % i for i in loss_list]))
                    self.log_file.flush()

                self.global_step += 1
            self.scheduler.step()

            metric_arr = []
            self.model.eval()
            rmsds = []
            with torch.no_grad():
                for batch in tqdm(self.valid_loader):
                    batch = self.to_device(batch, device)
                    loss, loss_list = self.model(batch['X'], batch['S'], batch['L'], batch['offsets'])
                    
                    _, xs, true_xs, __ = self.model.infer(batch, device)
                    rmsds.extend([compute_rmsd(xs[i], true_xs[i], True)for i in range(len(xs))])
                    metric_arr.append(loss.cpu().item())
                    self.valid_global_step += 1

            valid_metric = np.nanmean(metric_arr) if not use_rmsd else np.nanmean(rmsds)
            if valid_metric < self.best_valid_metric:
                self.patience = 40
                save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
                torch.save(self.model, save_path)
                torch.save(self.model, os.path.join(self.model_dir, f'best.ckpt'))
                self.best_valid_metric = valid_metric
                self.best_valid_epoch = self.epoch
            else:
                self.patience -= 1

            print("\033[0;30;43m {} | Epoch: {} | Val Loss: {:.5f}, Best Val: {:.5f}, Best Epoch: {}\033[0m".format(time.strftime("%Y-%m-%d %H-%M-%S"), self.epoch, valid_metric.item(), self.best_valid_metric, self.best_valid_epoch))
            self.log_file.write("{} | Epoch: {} | Val Loss: {:.5f}, Best Val: {:.5f}, Best Epoch: {}\n".format(time.strftime("%Y-%m-%d %H-%M-%S"), self.epoch, valid_metric.item(), self.best_valid_metric, self.best_valid_epoch))
            self.log_file.flush()

            self.epoch += 1
            if self.patience <= 0:
                print(f'Early Stopping!')
                break