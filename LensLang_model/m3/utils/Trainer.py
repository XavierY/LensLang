import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import *
from tqdm import tqdm
from utils.metrics import *
import copy


class Trainer():
    def __init__(
        self,
        model,
        device,
        lr,
        dataloaders,
        save_param_path,
        writer,
        early_stop,
        epoches,
        model_name,
        save_predict_result_path,
        beta_c,
        beta_n,
        scheduler_option=False,
        save_threshold=0.7,
        start_epoch=0,
        # 现在只保留 mode，用于命名和日志
        mode=None,
    ):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.early_stop = early_stop
        self.save_threshold = save_threshold
        self.writer = writer
        self.scheduler_option = scheduler_option
        self.beta_c = beta_c
        self.beta_n = beta_n

        # ---- mode 信息 ----
        self.mode = mode if mode is not None else "train"

        # ---- 模型参数保存路径 ----
        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            os.makedirs(save_param_path, exist_ok=True)
            self.save_param_path = save_param_path

        # ---- 结果保存根目录 ----
        if os.path.exists(save_predict_result_path):
            self.save_predict_result_path = save_predict_result_path
        else:
            os.makedirs(save_predict_result_path, exist_ok=True)
            self.save_predict_result_path = save_predict_result_path

        # ---- 统一时间戳与运行目录 ----
        # 同一轮 run 的 csv & log 共用一个 timestamp
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(
            self.save_predict_result_path,
            f"{self.mode}_{self.timestamp}"
        )
        os.makedirs(self.run_dir, exist_ok=True)

        self.lr = lr
        self.CEloss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=5e-5
        )

        if scheduler_option:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=1,
                min_lr=1e-6,
                verbose=True
            )

        # ---- 写参数 log 文件 ----
        log_path = os.path.join(
            self.run_dir,
            f"runlog_{self.mode}_{self.timestamp}.txt"
        )
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("Trainer init arguments summary\n")
                f.write(f"timestamp: {self.timestamp}\n")
                f.write(f"mode: {self.mode}\n")
                f.write(f"model_name: {self.model_name}\n")
                f.write(f"device: {self.device}\n")
                f.write(f"lr: {self.lr}\n")
                f.write(f"beta_c: {self.beta_c}\n")
                f.write(f"beta_n: {self.beta_n}\n")
                f.write(f"early_stop: {self.early_stop}\n")
                f.write(f"num_epochs: {self.num_epochs}\n")
                f.write(f"save_param_path: {self.save_param_path}\n")
                f.write(f"save_predict_result_path(root): {self.save_predict_result_path}\n")
                f.write(f"run_dir: {self.run_dir}\n")
                f.write(f"scheduler_option: {self.scheduler_option}\n")
                f.write(f"start_epoch: {self.start_epoch}\n")
                f.write(f"save_threshold: {self.save_threshold}\n")
                # dataloader 信息简单记录一下
                if isinstance(self.dataloaders, dict):
                    for k, v in self.dataloaders.items():
                        try:
                            size = len(v.dataset)
                        except Exception:
                            size = "unknown"
                        f.write(f"dataloader[{k}] size: {size}\n")
        except Exception as e:
            print(f"[WARN] Failed to write Trainer runlog: {e}")

    def train(self):
        since = time.time()
        self.model.cuda()

        best_model_wts_test = copy.deepcopy(self.model.state_dict())
        best_f1_test = 0.0
        best_epoch_test = 0
        is_earlystop = False

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch, self.start_epoch + self.num_epochs - 1))
            print('-' * 50)

            # ----------------- TRAIN -----------------
            self.model.train()
            print('-' * 10)
            print('TRAIN')
            print('-' * 10)
            running_loss = 0.0
            tpred = []
            tlabel = []

            for batch in tqdm(self.dataloaders['train']):
                self.optimizer.zero_grad()
                batch_data = batch
                for k, v in batch_data.items():
                    if k != 'vid':
                        batch_data[k] = v.cuda()
                labels = batch_data['label']
                outputs, output_content, output_narative = self.model(**batch_data)

                loss = (
                    self.CEloss(outputs, labels)
                    + self.beta_c * self.CEloss(output_content, labels)
                    + self.beta_n * self.CEloss(output_narative, labels)
                )

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * labels.size(0)
                tpred.extend(torch.max(outputs, 1)[1].tolist())
                tlabel.extend(labels.tolist())

            epoch_loss = running_loss / len(self.dataloaders['train'].dataset)
            print('Train Loss: {:.4f} '.format(epoch_loss))
            results = metrics(tlabel, tpred)
            print(results)

            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            self.writer.add_scalar('Acc/train', results['acc'], epoch)
            self.writer.add_scalar('F1/train', results['f1'], epoch)

            # ----------------- VAL -----------------
            self.model.eval()
            print('-' * 10)
            print('VAL')
            print('-' * 10)
            val_loss = 0.0
            val_tpred = []
            val_tlabel = []

            for batch in tqdm(self.dataloaders['val']):
                batch_data = batch
                for k, v in batch_data.items():
                    if k != 'vid':
                        batch_data[k] = v.cuda()
                labels = batch_data['label']
                with torch.no_grad():
                    outputs, output_content, output_narative = self.model(**batch_data)
                    loss = (
                        self.CEloss(outputs, labels)
                        + self.beta_c * self.CEloss(output_content, labels)
                        + self.beta_n * self.CEloss(output_narative, labels)
                    )

                val_loss += loss.item() * labels.size(0)
                val_tpred.extend(torch.max(outputs, 1)[1].tolist())
                val_tlabel.extend(labels.tolist())

            epoch_loss_val = val_loss / len(self.dataloaders['val'].dataset)
            print('Val Loss: {:.4f} '.format(epoch_loss_val))
            results_val = metrics(val_tlabel, val_tpred)
            print(results_val)

            if self.scheduler_option:
                self.scheduler.step(epoch_loss_val)

            if results_val['f1'] > best_f1_test:
                best_f1_test = results_val['f1']
                best_epoch_test = epoch
                best_model_wts_test = copy.deepcopy(self.model.state_dict())
                if best_f1_test > self.save_threshold:
                    ckpt_name = (
                        self.model_name
                        + "_val_"
                        + str(best_epoch_test)
                        + "_{0:.4f}".format(best_f1_test)
                    )
                    ckpt_path = os.path.join(self.save_param_path, ckpt_name)
                    torch.save(best_model_wts_test, ckpt_path)
                    print("saved " + ckpt_path)
            else:
                if epoch - best_epoch_test >= self.early_stop - 1:
                    is_earlystop = True
                    print("early stop at epoch " + str(epoch))

        time_elapsed = time.time() - since
        print(
            'Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best model on val: epoch" + str(best_epoch_test) + "_" + str(best_f1_test))
        ckp_path = os.path.join(
            self.save_param_path,
            self.model_name
            + "_val_"
            + str(best_epoch_test)
            + "_{0:.4f}".format(best_f1_test),
        )

        return ckp_path

    def test(self, ckp_path):
        self.model.load_state_dict(torch.load(ckp_path))
        since = time.time()
        self.model.cuda()
        self.model.eval()
        pred = []
        label = []
        vid = []

        for batch in tqdm(self.dataloaders['test']):
            with torch.no_grad():
                batch_data = batch
                for k, v in batch_data.items():
                    if k != 'vid':
                        batch_data[k] = v.cuda()
                labels = batch_data['label']
                outputs, output_content, output_narative = self.model(**batch_data)
                label.extend(labels.tolist())
                pred.extend(torch.max(outputs, 1)[1].tolist())
                vid.extend(batch_data['vid'])

        time_elapsed = time.time() - since
        print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # ---------- wrong 列 + CSV 命名 ----------
        label_arr = np.array(label)
        pred_arr = np.array(pred)
        wrong = (pred_arr != label_arr).astype(int).tolist()

        result = pd.DataFrame({
            'vid': vid,
            'label': label,
            'pred': pred,
            'wrong': wrong
        })

        csv_name = f"label_{self.mode}_{self.timestamp}.csv"
        csv_path = os.path.join(self.run_dir, csv_name)

        # 用默认逗号分隔，utf-8-sig 方便 Excel 识别
        result.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Saved test result to: {csv_path}")

        print(get_confusionmatrix_fnd(pred_arr, label_arr))
        print(metrics(label, pred))

        return metrics(label, pred)


class Inferencer():
    def __init__(
        self,
        model,
        device,
        model_name,
        dataloader,
        save_predict_result_path,
        mode=None,
    ):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.dataloader = dataloader
        self.mode = mode if mode is not None else "infer"

        if os.path.exists(save_predict_result_path):
            self.save_predict_result_path = save_predict_result_path
        else:
            os.makedirs(save_predict_result_path, exist_ok=True)
            self.save_predict_result_path = save_predict_result_path

        # 统一时间戳与 run 目录
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(
            self.save_predict_result_path,
            f"{self.mode}_{self.timestamp}"
        )
        os.makedirs(self.run_dir, exist_ok=True)

        # 写参数 log
        log_path = os.path.join(
            self.run_dir,
            f"runlog_{self.mode}_{self.timestamp}.txt"
        )
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("Inferencer init arguments summary\n")
                f.write(f"timestamp: {self.timestamp}\n")
                f.write(f"mode: {self.mode}\n")
                f.write(f"model_name: {self.model_name}\n")
                f.write(f"device: {self.device}\n")
                f.write(f"save_predict_result_path(root): {self.save_predict_result_path}\n")
                f.write(f"run_dir: {self.run_dir}\n")
                try:
                    size = len(self.dataloader.dataset)
                except Exception:
                    size = "unknown"
                f.write(f"dataloader size: {size}\n")
        except Exception as e:
            print(f"[WARN] Failed to write Inferencer runlog: {e}")

    def inference(self, ckp_path):
        self.model.load_state_dict(torch.load(ckp_path), strict=False)
        since = time.time()
        self.model.cuda()
        self.model.eval()

        label = []
        vid = []
        pred = []

        for batch in tqdm(self.dataloader):
            with torch.no_grad():
                batch_data = batch
                for k, v in batch_data.items():
                    if k != 'vid':
                        batch_data[k] = v.cuda()
                labels = batch_data['label']
                outputs, output_content, output_narative = self.model(**batch_data)
                label.extend(labels.tolist())
                pred.extend(torch.max(outputs, 1)[1].tolist())
                vid.extend(batch_data['vid'])

        time_elapsed = time.time() - since
        print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # ---------- wrong 列 + CSV 命名 ----------
        label_arr = np.array(label)
        pred_arr = np.array(pred)
        wrong = (pred_arr != label_arr).astype(int).tolist()

        result = pd.DataFrame({
            'vid': vid,
            'label': label,
            'pred': pred,
            'wrong': wrong
        })

        csv_name = f"label_{self.mode}_{self.timestamp}.csv"
        csv_path = os.path.join(self.run_dir, csv_name)
        result.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Saved inference result to: {csv_path}")

        print(get_confusionmatrix_fnd(pred_arr, label_arr))
        print(metrics(label, pred))

        return metrics(label, pred)