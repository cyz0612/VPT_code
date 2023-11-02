from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pad_len = 150

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
        }
        self.args.seq_len = pad_len
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        self.args.seq_len = 7
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _get_csv(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def vali_list(self, val_list, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for m in range(len(val_list)):
                _,vali_loader = val_list[m]
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    if i == 0:
                        batch_all_i = batch_x
                        batch_mark_all = batch_x_mark
                    else:
                        batch_all_i = torch.concat((batch_all_i,batch_x),axis=1)
                    temp = torch.zeros((batch_all_i.shape[0],pad_len-batch_all_i.shape[1],batch_all_i.shape[2])).to(self.device)
                    temp_mark = torch.zeros((batch_mark_all.shape[0],pad_len-batch_mark_all.shape[1],batch_mark_all.shape[2])).to(self.device)

                    input_batch = torch.concat((temp,batch_all_i),axis=1)
                    input_batch_mark = torch.concat((temp_mark,batch_mark_all),axis=1)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, input_batch_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, input_batch_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(input_batch)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(input_batch, input_batch_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(input_batch, input_batch_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    loss = criterion(pred, true)

                    total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        all_csv = os.listdir(self.args.root_path)
        ds_len = len(all_csv)

        trian_csv = all_csv[:int(0.7*ds_len)]
        val_csv = all_csv[int(0.7*ds_len):int(0.8*ds_len)]
        test_csv = all_csv[int(0.8*ds_len):]

        train_loader_list = []
        for csv_file in trian_csv:
            self.args.data_path = csv_file
            aaa = pd.read_csv(os.path.join(self.args.root_path,csv_file))
            print("trian length:",len(aaa))
            train_data, train_loader = self._get_data(flag='train')
            train_loader_list.append([train_data, train_loader])

        val_loader_list = []
        for csv_file in val_csv:
            self.args.data_path = csv_file
            vali_data, vali_loader = self._get_data(flag='val')
            val_loader_list.append([vali_data, vali_loader])

        test_loader_list = []
        for csv_file in test_csv:
            self.args.data_path = csv_file
            test_data, test_loader = self._get_data(flag='test')
            test_loader_list.append([test_data, test_loader])
        
        self.test_loader_list = test_loader_list
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        self.train_loss_list = []
        self.vali_loss_list = []
        for epoch in range(self.args.train_epochs):
            print("-------------------trianing_epoch:",epoch)
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for j in range(len(train_loader_list)):
                train_data, train_loader = train_loader_list[j][0],train_loader_list[j][1]
                print(">>>trianing file:",all_csv[j])


                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    # print("batch_x:",batch_x.shape,
                    #       "batch_y:",batch_y.shape,
                    #       "batch_x_mark:",batch_x_mark.shape,
                    #       "batch_y_mark:",batch_y_mark.shape)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    
                
                    if i == 0:
                        batch_all = batch_x
                        batch_mark_all = batch_x_mark
                    else:
                        batch_all = torch.concat((batch_all,batch_x),axis=1)
                    temp = torch.zeros((batch_all.shape[0],pad_len-batch_all.shape[1],batch_all.shape[2])).to(self.device)
                    temp_mark = torch.zeros((batch_mark_all.shape[0],pad_len-batch_mark_all.shape[1],batch_mark_all.shape[2])).to(self.device)

                    input_batch = torch.concat((temp,batch_all),axis=1)
                    input_batch_mark = torch.concat((temp_mark,batch_mark_all),axis=1)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(input_batch)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(input_batch, input_batch_mark, dec_inp, batch_y_mark)[0]
                                
                            else:
                                outputs = self.model(input_batch, input_batch_mark, dec_inp, batch_y_mark, batch_y)
                        # print(outputs.shape,batch_y.shape)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        # print(outputs.shape,batch_y.shape)
                        # print("!!!!!!!!!!output:",outputs)
                        train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                        
                    if self.args.lradj == 'TST':
                        adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                        scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)
            vali_loss = self.vali_list(val_loader_list, criterion)
            test_loss = self.vali_list(test_loader_list, criterion)

            self.train_loss_list.append(train_loss)
            self.vali_loss_list.append(vali_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    
                    # print(input.shape)
                    # print(input[0, :, -1].shape)
                    # print(pred[0, :, -1].shape)

                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # gt = np.concatenate((np.array(inputx)[0, :, -1], np.array(trues)[0, :, -1]), axis=0)
                    # pd = np.concatenate((np.array(inputx)[0, :, -1], np.array(preds)[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.jpg'))
    
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def test_list(self, setting, test=0):
        # test_data, test_loader = self._get_data(flag='test')
        all_csv = os.listdir(self.args.root_path)
        ds_len = len(all_csv)
        test_csv = all_csv[int(0.8*ds_len):]
        test_loader_list = []
        for csv_file in test_csv:
            self.args.data_path = csv_file
            test_data, test_loader = self._get_data(flag='test')
            test_loader_list.append([test_data, test_loader])
        
        self.test_loader_list = test_loader_list

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for n in range(len(self.test_loader_list)):
                _,test_loader = self.test_loader_list[n]
                batch_all = 0
                pd_list = []
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    if i == 0:
                        batch_all_i = batch_x
                        batch_mark_all = batch_x_mark
                    else:
                        batch_all_i = torch.concat((batch_all_i,batch_x),axis=1)
                    temp = torch.zeros((batch_all_i.shape[0],pad_len-batch_all_i.shape[1],batch_all_i.shape[2])).to(self.device)
                    temp_mark = torch.zeros((batch_mark_all.shape[0],pad_len-batch_mark_all.shape[1],batch_mark_all.shape[2])).to(self.device)

                    input_batch = torch.concat((temp,batch_all_i),axis=1)
                    input_batch_mark = torch.concat((temp_mark,batch_mark_all),axis=1)
                    
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(input_batch)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(input_batch, input_batch_mark, dec_inp, batch_y_mark)[0]

                            else:
                                outputs = self.model(input_batch, input_batch_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    # print(outputs.shape,batch_y.shape)
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                    true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                    preds.append(pred)
                    trues.append(true)
                    inputx.append(batch_x.detach().cpu().numpy())
                    if i % 1 == 0:
                        input = batch_x.detach().cpu().numpy()
                        
                        if i == 0:
                            batch_all = input
                        else:
                            batch_all = np.concatenate((batch_all,input),axis=1)
                        # print(input.shape)
                        # print(input[0, :, -1].shape)
                        # print(pred[0, :, -1].shape)

                        # gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        pd1 = np.concatenate((batch_all[0, :, -1], pred[0, :, -1]), axis=0)
                        pd_list.append(pd1)
                
                batch_all = np.concatenate((batch_all[0, :, -1],true[0, :, -1]),axis=0)
                gt = batch_all
                # gt = batch_all[0, :, -1]
                for iii in range(len(pd_list)):
                    pd = pd_list[iii]
                    visual(gt,iii+1,pd, os.path.join(folder_path, str(n)+str(iii) + '.jpg'))

        plt.figure()
        plt.plot(range(len(self.train_loss_list)), self.train_loss_list,label = "train loss")
        plt.plot(range(len(self.vali_loss_list)), self.vali_loss_list,label = "validation loss")
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title("Loss Curve")
        plt.savefig(folder_path+"train_loss_curve.jpg",dpi=300)
        plt.show()

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
