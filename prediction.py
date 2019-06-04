import app
from main import args, fields, include_test, evaluator, log_printer, batch_size_fn_zoo
import os
import torch
from torch.autograd import Variable
import numpy as np

print(os.getpid())

class Tester(app.TestAPP):
    def predict(self, dataset_iter, dataset_name):
        with torch.no_grad():
            small_batch_size = 32
            self.model.eval()
            self.test_iter.init_epoch()
            test_result = []
            test_loss = 0
            if self.output_parser != None:
                os.makedirs(self.args.result_path, exist_ok=True)
                fout = open(os.path.join(args.result_path, "{}_{}_{}_{}".format(dataset_name,
                                                                                args.train_dataset,
                                                                                args.model_type,
                                                                                args.gating_source)), 'w')
            for test_batch_idx, test_batch in enumerate(dataset_iter):
                small_batch = (test_batch.QUESTION.size(0) - 1) // small_batch_size + 1
                logit = []
                feature = []
                for i in range(small_batch):
                    if i == small_batch - 1:
                        sent1 = test_batch.QUESTION[small_batch_size * i:]
                        sent2 = test_batch.ANSWER[small_batch_size * i:]
                        ext = test_batch.IRFEATURE[small_batch_size * i:]
                    else:
                        sent1 = test_batch.QUESTION[small_batch_size * i:small_batch_size * (i + 1)]
                        sent2 = test_batch.ANSWER[small_batch_size * i:small_batch_size * (i + 1)]
                        ext = test_batch.IRFEATURE[small_batch_size * i: small_batch_size * (i + 1)]
                    if self.model.config.ext_feats:
                        test_output_ = self.model(sent1, sent2, ext)
                    else:
                        test_output_ = self.model(sent1, sent2, None)
                    logit.append(test_output_[0])
                    feature.append(test_output_[1])
                test_output = torch.cat(logit, dim=0)
                test_feature = torch.cat(feature, dim=0)
                if dataset_name != "train":
                    test_result.append((test_output, test_batch))
                if self.output_parser != None:
                    self.output_parser(test_batch, test_output, test_feature, fout)
            if dataset_name != "train":
                test_metrics = self.evaluator(dataset_name, test_result)
                self.log_printer(dataset_name, loss=test_loss, metrics=test_metrics)
    def test(self):
        #self.predict(dataset_iter=self.train_iter, dataset_name='train')
        self.predict(dataset_iter=self.valid_iter, dataset_name='valid')
        self.predict(dataset_iter=self.test_iter, dataset_name='test')

def output_parser(batch, logit, feature, fout):
    for QID, QSEQ, fea, score, label in zip(batch.QID.cpu().data.numpy(), batch.QSEQ.cpu().data.numpy(),
                                            feature.cpu().data.numpy(), logit.cpu().data.numpy(), batch.LABEL.cpu().data.numpy()):
        feature_vector = " ".join([str(num) for num in fea])
        fout.write("\t".join([str(QID), str(QSEQ), feature_vector, str(np.exp(float(score[1]))), str(label)]) + "\n")


tester = Tester(args=args, fields=fields, include_test=include_test,
                batch_size_fn_train=batch_size_fn_zoo['train{}'.format(args.train_dataset)],
                batch_size_fn_valid=batch_size_fn_zoo['valid{}'.format(args.train_dataset)],
                batch_size_fn_test=batch_size_fn_zoo['test{}'.format(args.train_dataset)]
                )
tester.prepare(evaluator=evaluator, log_printer=log_printer, output_parser=output_parser)
tester.test()