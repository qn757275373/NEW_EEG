from PSD_dataset import *
# from code.logger import *
from configure import opt
from torch.utils.data import DataLoader
# from model_LSTM.model import *
from model import *
from utils import *
import torch.nn.functional as F
# from logger.logg import *
import torch
from tqdm import tqdm, trange
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from tensorboardX import SummaryWriter
from sklearn.svm import SVC
import time
from torchsummary import summary


modelname = 'TEarea_FT_Transformer'
save_weights_path = '/home/mly/PycharmProjects/NEW_EEG/code/' + modelname + '/checkpoints/new'
save_weights_path_best = '/home/mly/PycharmProjects/NEW_EEG/code/' + modelname + '/checkpoints/new/best'
mylog_save_path = '/home/mly/PycharmProjects/NEW_EEG/code/' + modelname + '/log'
SummaryWriter_path = '/home/mly/PycharmProjects/NEW_EEG/code/' + modelname + '/run_new'

# save weights
def saveWeights(train_acc, val_acc, test_acc, epoch, model, optimizer):
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
    for root, dir, files in os.walk("checkpoints"):
        if len(files) != 0:
            for file in files:
                # file = file.split('_')[5]
                # test = test_acc
                # if float(file.split('_')[5]) < test_acc:    # test_acc小于当前acc的模型被替换，大于当前acc的模型仍然存在,如果当前acc比所有模型都好，那所有模型都会被删除，只剩当前最好的acc
                if len(file.split('_')) > 5 and float(file.split('_')[5]) < test_acc:
                    os.remove(os.path.join(root, file))
                    save_file = save_weights_path + 'ImageNet_TransformerEncoder_%d_%.4f_%.4f_%.4f_weights.pth' % (
                        epoch, train_acc, val_acc, test_acc)
                    torch.save(state, save_file)
                    print('The model parameters have been removed before and saved successed')
        else:
            save_file = save_weights_path + 'ImageNet_TransformerEncoder_%d_%.4f_%.4f_%.4f_weights.pth' % (
                epoch, train_acc, val_acc, test_acc)
            torch.save(state, save_file)
            print('The model parameters have been saved successed')



def main(args):
    accuracy_sum = 0.0
    if opt.log_note is None:
        opt.log_note = str(datetime.datetime.now())
        opt.log_note = opt.log_note.replace(' ', '-').replace(':', '_').split('.')[0]
    log_save_path = mylog_save_path + '/' + opt.log_note
    cycle = 1
    for i in range(cycle):
        torch.cuda.synchronize()  # 增加同步操作
        start = time.time()
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        dataset = EEGDataset(opt.eeg_dataset, opt)
        # Create loaders
        # 数据加载器。组合了一个数据集和采样器，并提供关于数据的迭代器。
        loaders = {
            split: DataLoader(Splitter(dataset, split_path=opt.splits_path, split_num=opt.split_num, split_name=split),
                              batch_size=opt.batch_size, drop_last=True, shuffle=True) for split in
            ["train", "val", "test"]}
        writer = SummaryWriter(logdir=SummaryWriter_path,flush_secs=2)
        # model = LSTM_Attention_Skip(440, 128, 128, 1, 40, True, 2)
        model = Transformer().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.init_lr)
        last_lr = opt.init_lr
        model = model.cuda()
        # results=summary(model, [(6,128),(440,128)], col_names=["kernel_size","output_size","num_params"],)

        def adjust_learning_rate(optimizer, epoch, lr):
            """Sets the learning rate to the initial LR decayed by 10 every 60 epochs"""
            lr *= (0.98 ** (epoch // 100))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        lr_init = optimizer.param_groups[0]['lr']
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=opt.lr_decay_eps,
        #                                                           patience=opt.lr_decay_patience, verbose=True,
        #                                                           threshold=opt.lr_decay_th,
        #                                                           threshold_mode='abs')
        # 加载预训练模型
        ckpt = torch.load('/home/mly/PycharmProjects/NEW_EEG/code/TEarea_FT_Transformer/checkpoints/test_acc_0_9939516129032258.pth')
        model.load_state_dict(ckpt['net'])
        classifier = SVC(C=0.5)
        # logger = Logger(opt.run_save_dir)
        epoch = 0
        best_epoch = 1
        train_count = 0
        glo_best_test_acc = 0.97
        while epoch < 1400:
            epoch += 1
            opt.current_epoch = epoch
            adjust_learning_rate(optimizer, epoch, lr_init)
            cur_lr = optimizer.param_groups[0]['lr']

            # Initialize loss/accuracy variables
            losses = {"train": 0, "val": 0, "test": 0}
            accuracies = {"train": 0, "val": 0, "test": 0}
            counts = {"train": 0, "val": 0, "test": 0}
            # adjust_lr(epoch, optimizer, opt)
            # Process each split
            for split in ("train", "val", "test"):
                # Set network mode
                if split == "train":
                    model.train()
                    torch.set_grad_enabled(True)
                else:
                    model.eval()
                    torch.set_grad_enabled(False)
                # Process all split batches
                for i, (fre_input, time_input, target) in enumerate(tqdm(loaders[split])):#tqdm 进度条
                    # Check CUDA
                    fre_input = fre_input.cuda(non_blocking=True)
                    time_input = time_input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    # Forward
                    # input = torch.unsqueeze(input, 1)
                    output, _ = model(fre_input, time_input)
                    # feature_output = model.feature
                    loss = F.cross_entropy(output, target)
                    losses[split] += loss.item()
                    # Compute accuracy
                    _, pred = output.data.max(1)

                    # use classifier
                    # feature_output_c = feature_output.cpu().detach().numpy()
                    # # output_c = output.cpu().detach().numpy()
                    # target_c = target.cpu().detach().numpy()
                    # if split == "train":
                    #     classifier.fit(feature_output_c, target_c)
                    #     predict = classifier.predict(feature_output_c)
                    #     train_acc = classifier.score(feature_output_c, target_c)
                    # elif split == "val":
                    #     predict = classifier.predict(feature_output_c)
                    #     val_acc = classifier.score(feature_output_c, target_c)
                    # else:
                    #     test_acc = classifier.score(feature_output_c, target_c)
                    # pred = torch.Tensor(predict)
                    # pred = pred.cuda(non_blocking=True)
                    # target.data = target.data.float()

                    correct = pred.eq(target.data).sum().item()
                    accuracy = correct / time_input.data.size(0)
                    accuracies[split] += accuracy
                    counts[split] += 1
                    # Backward and optimize
                    if split == "train":
                        # 清空过往梯度
                        optimizer.zero_grad()
                        # 反向传播，计算当前梯度
                        loss.backward()
                        # 根据梯度更新网络参数
                        optimizer.step()
                # Print info at the end of the epoch

            opt.train_loss = losses['train'] / counts['train']
            opt.train_acc = accuracies['train'] / counts['train']
            opt.val_loss = losses['val'] / counts['val']
            opt.val_acc = accuracies['val'] / counts['val']
            opt.test_loss = losses['test'] / counts['test']
            opt.test_acc = accuracies['test'] / counts['test']


            # logger.scalar_summary('train_loss', opt.train_loss, epoch)
            # logger.scalar_summary('train_acc', opt.train_acc, epoch)
            # logger.scalar_summary('val_loss', opt.val_acc, epoch)
            # logger.scalar_summary('val_acc', opt.val_acc, epoch)
            # logger.scalar_summary('test_loss', opt.test_loss, epoch)
            # logger.scalar_summary('test_acc', opt.test_acc, epoch)
            # save_model(model, opt.all_model_dir, f'epoch:{epoch}.pth',optimizer, opt, epoch, lr_scheduler, True)
            if epoch % 1 == 0:
                save_model(model, save_weights_path, f'epoch:{epoch}.pth',optimizer, opt, epoch, True)

            writer.add_scalar('train_loss', opt.train_loss, global_step=epoch)
            writer.add_scalar('train_acc', opt.train_acc, global_step=epoch)
            writer.add_scalar('val_loss', opt.val_loss, global_step=epoch)
            writer.add_scalar('val_acc', opt.val_acc, global_step=epoch)
            writer.add_scalar('test_loss', opt.test_loss, global_step=epoch)
            writer.add_scalar('test_acc', opt.test_acc, global_step=epoch)

            if opt.best_test_acc is None:
                opt.best_test_acc = opt.test_acc
                opt.best_train_acc = opt.train_acc
            if opt.test_acc > opt.best_test_acc:
                opt.best_test_acc = opt.test_acc
                opt.best_val_acc = opt.val_acc
                opt.best_train_acc = opt.train_acc
                best_epoch = epoch
                save_and_clean_model(model, save_weights_path_best, f'test_acc_{str(opt.test_acc).replace(".", "_")}.pth')
            print_to_log_file(log_save_path, f"Epoch {epoch}: "
                                             f"TrL={opt.train_loss:.4f}, "
                                             f"TrA={opt.train_acc:.4f}, "
                                             f"TvL={opt.val_loss:.4f}, "
                                             f"TvA={opt.val_acc:.4f}, "
                                             f"TeL={opt.test_loss:.4f}, "
                                             f"TeA={opt.test_acc:.4f}")
            if opt.train_acc >= 0.99:
                train_count += 1
            # if train_count > 80:
            #     break

            if opt.test_acc >= glo_best_test_acc:
                saveWeights(opt.train_acc, opt.val_acc, opt.test_acc, epoch, model, optimizer)
                glo_best_test_acc = max(opt.test_acc, opt.best_test_acc)

        accuracy_sum += opt.best_test_acc
        print_to_log_file(log_save_path, f"best_epoch: {best_epoch}, "
                                         f"best_train_acc:{opt.best_train_acc:.4f}, "
                                         f"best_val_acc:{opt.best_val_acc:.4f}, "
                                         f"best_test_acc:{opt.best_test_acc:.4f}")
        end = time.time()
        elapsed_time = end - start
        print_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print_to_log_file(log_save_path, f"time: {print_time}")
        torch.cuda.synchronize()  # 增加同步操作
        opt.best_test_acc = None
        writer.close()

    print_to_log_file(log_save_path, 'average:', accuracy_sum / cycle)


def classifier_training(features, labels):
    '''
    训练分类器
    '''
    classifier = SVC(C=0.5)
    # classifier = MLPClassifier()
    # classifier = RandomForestClassifier(n_jobs=4, criterion='entropy', n_estimators=70, min_samples_split=5)
    # classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    # classifier = ExtraTreesClassifier(n_jobs=4,  n_estimators=100, criterion='gini', min_samples_split=10,
    #                        max_features=50, max_depth=40, min_samples_leaf=4)
    # classifier = GaussianNB()
    classifier.fit(features, labels)
    predict = classifier.predict(features)
    prediction = predict.tolist()
    return prediction




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a network for viable tumor segmentation')
    parser.add_argument('--bs', default=8, type=int)
    parser.add_argument('--gpu', default=1, type=int)
    args = parser.parse_args()


    main(args)
