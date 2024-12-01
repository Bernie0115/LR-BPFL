import os
import time
import h5py
import torch
import numpy as np
from flcore.servers.serverbase import Server
# from flcore.clients.clientMetabayes import clientMetaBAYES
from flcore.clients.clientMetabayes_joint import clientMetaBAYES
from torchmetrics.functional.classification import multiclass_calibration_error


class FedMetaBayes(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.rank = args.rank

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMetaBAYES)
        self.selected_clients = None

        self.decay = 100

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):

            s_t = time.time()

            self.selected_clients = self.select_clients()
            self.send_models()

            # if i == self.decay:
            #     for client in self.clients:
            #         # lr = client.optimizer_W.param_groups[0]['lr'] * 0.5
            #         for param_group in client.optimizer_W.param_groups:
            #             param_group['lr'] = 0.001

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate model with multiple local updates")
                self.evaluate(i)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientMetaBAYES)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate(self.global_rounds + 1)

    # evaluate selected clients
    def evaluate(self, global_round):

        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])

        self.writer.add_scalar("Train_loss", train_loss, global_round)
        self.writer.add_scalar("Test_acc", test_acc, global_round)

        test_ece = multiclass_calibration_error(stats[3], stats[4], num_classes=self.num_classes, n_bins=15, norm='l1')
        test_mce = multiclass_calibration_error(stats[3], stats[4], num_classes=self.num_classes, n_bins=15, norm='max')

        self.rs_test_acc.append(test_acc)
        self.rs_test_ece.append(test_ece)
        self.rs_test_mce.append(test_mce)

        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Test ECE: {:.4f}".format(test_ece))
        print("Test MCE: {:.4f}".format(test_mce))


    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_prob = []
        tot_true = []
        for c in self.clients:
            ct, ns, prob, true = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_prob.append(prob)
            tot_true.append(true)
            num_samples.append(ns)

        tot_prob = torch.cat(tot_prob, dim=0)
        tot_true = torch.cat(tot_true, dim=0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_prob, tot_true

    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_prob = []
        tot_true = []
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            ct, ns, prob, true = client.test_metrics(update_step=self.fine_tuning_epoch)
            tot_correct.append(ct * 1.0)
            tot_prob.append(prob)
            tot_true.append(true)
            num_samples.append(ns)

        tot_prob = torch.cat(tot_prob, dim=0)
        tot_true = torch.cat(tot_true, dim=0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_prob, tot_true

    def save_results(self):
        loca = time.time()
        loca = time.strftime("%Y_%m_%d_%H_%M_%S")

        algo = (self.dataset + "_" + self.algorithm + '_' +str(self.learning_rate) + '_' + str(self.num_clients) + '_'
                + str(self.join_ratio)) + "_" + str(self.rank)
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc):
            algo = algo + "_" + str(loca)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_ece', data=self.rs_test_ece)
                hf.create_dataset('rs_test_mce', data=self.rs_test_mce)

