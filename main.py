import os
import app
import evaluation
import logger
import time
from torchtext import data
import torch
from model_attn import Attention
from model_attn_dot import AttentionDot
from model_qac import FastDynamic
from baseline import SM
import json

print(os.getpid())

class Args(app.ArgParser):
    def __init__(self):
        super(Args, self).__init__(description="Twitter Search", batch_size=1, dev_every=30, log_every=1, patience=1000,
                                   dataset_path="data")
        self.parser.add_argument('--word_embed_dim', type=int, default=300)
        self.parser.add_argument('--ext_feats', action='store_true', default=False,
                            help='use sparse features (default: false)')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability (default: 0.5)')
        self.parser.add_argument('--output_channel', type=int, default=350)
        self.parser.add_argument('--hidden_size', type=int, default=350)
        self.parser.add_argument('--attn_hidden', type=int, default=300)
        self.parser.add_argument('--hidden_layer_units', type=int, default=100)
        self.parser.add_argument('--kernel_size', type=int, default=2)
        self.parser.add_argument('--vector_cache', type=str,
                                 default="data/twitter.glove.pt",
                                 help="word embedding file, pt format")
        self.parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.001)')
        self.parser.add_argument('--weighted_loss', default=False, action="store_true")
        self.parser.add_argument('--tensorboard', type=str, default='logs')
        self.parser.add_argument('--train_dataset', type=str, default='134')
        self.parser.add_argument('--model_type', default="attn", type=str)

        ## For ablation experiment
        self.parser.add_argument('--gating_source', default="embed", type=str)

        ## For MPCNN
        self.parser.add_argument('--max_window_size', type=int, default=3,
                                 help='windows sizes will be [1,max_window_size] and infinity (default: 300)')
        self.parser.add_argument('--holistic_filters', type=int, default=300,
                                 help='number of holistic filters (default: 300)')
        self.parser.add_argument('--per_dim_filters', type=int, default=20,
                                 help='number of per-dimension filters (default: 20)')
        self.parser.add_argument('--small_batch_size', type=int, default=256)

        ## For BiMPM
        self.parser.add_argument('--n_word_dim', type=int, default=300)
        self.parser.add_argument('--n_perspectives', type=int, default=20)
        self.parser.add_argument('--n_hidden_units', type=int, default=100)
        self.parser.add_argument('--bimpm_dropout', type=float, default=0.1)

        self.parser.add_argument('--tfidf_file', type=str, default="data/idf_unigram.json")
        self.parser.add_argument("--shuffle", action="store_true", default=False)

arg_parser = Args()
args = arg_parser.get_args()
args.batch_size = 200
args.train_txt = 'train{}.combb'.format(args.train_dataset)
args.valid_txt = 'valid{}.combb'.format(args.train_dataset)
args.test_txt = 'test{}.combb'.format(args.train_dataset)
print(args)

# Fields
QID = data.Field(batch_first=True, sequential=False, preprocessing=lambda x:int(x), use_vocab=False)
QSEQ = data.Field(batch_first=True, sequential=False, preprocessing=lambda x:int(x), use_vocab=False)
TEXT = data.Field(batch_first=True)
LABEL = data.Field(batch_first=True, sequential=False, unk_token=None)
TIME = data.Field(batch_first=True, sequential=False, use_vocab=False)
IRFEATURE = data.Field(batch_first=True, sequential=True, use_vocab=False, tensor_type=torch.FloatTensor,
                       postprocessing=data.Pipeline(lambda arr, _, train: [float(y) for y in arr]))

fields = [('QID', QID), ('QSEQ', QSEQ), ('QUESTION',TEXT), ('ANSWER',TEXT), ('LABEL',LABEL),
          ('TIME',TIME), ('IRFEATURE',IRFEATURE)]
include_test = [False, False, True, True, False, False, False]


# Hack batch_size_fn to make examples groups with query id
# def batch(data, batch_size, batch_size_fn=lambda new, count, sofar: count):
#     """Yield elements from data in chunks of batch_size."""
#     minibatch, size_so_far = [], 0
#     for ex in data:
#         minibatch.append(ex)
#         size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
#         if size_so_far == batch_size:
#             yield minibatch
#             minibatch, size_so_far = [], 0
#         elif size_so_far > batch_size:
#             yield minibatch[:-1]
#             minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
#     if minibatch:
#         yield minibatch
# According to this function, we will define our batch_zise_fn
# For twitter dataset, we want to group twitter with same query. So we need to know how many twitters in one query
# And then create a dynamic batch
# batch_size = 1, batch_size_fn : if reach batch_size, return 1, else return 0

batch_size_fn_zoo = {}


class batch_size_fn:
    def __init__(self, boundary):
        self.boundary = boundary
        print(boundary)

    def __call__(self, new, count, sofar):
        # Before create Batch, example's attribute is not Variable
        # Need to use preprocessing to convert it into int
        if new.QSEQ == self.boundary[new.QID]:
            return 200
        return 0


for fname in ["train{}".format(args.train_dataset),
              "valid{}".format(args.train_dataset),
              "test{}".format(args.train_dataset)]:
    fboundary = open("data/{}.boundaryb".format(fname))
    boundary = {}
    for line in fboundary.readlines():
        key, value = line.strip().split('\t')
        boundary[int(key)] = int(value)
    if args.shuffle:
        batch_size_fn_zoo[fname] = None
    else:
        batch_size_fn_zoo[fname] = batch_size_fn(boundary)

class criterion:
    # You need to do any modification to loss here
    # TODO: Might need to pass model parameters
    def __init__(self):
        if args.weighted_loss:
            print("Use Weighted Loss")
            if args.cuda:
                self.crit = torch.nn.NLLLoss(weight=torch.FloatTensor([0.1, 1]).cuda(args.gpu))
            else:
                self.crit = torch.nn.NLLLoss(weight=torch.FloatTensor([0.1, 1]))
        else:
            self.crit = torch.nn.NLLLoss()


    def __call__(self, output, label):
        # return loss
        return self.crit(output[0], label)


class optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.SGD(parameter, lr = config.lr, weight_decay=1e-4, momentum=0.9)
        l = lambda epoch: 0.75 ** (epoch // 5)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def zero_grad(self):
        self.optim.zero_grad()

    def step(self):
        self.optim.step()

    def schedule(self):
        pass
        self.scheduler.step()
        print("learning rate : ", self.scheduler.get_lr(), self.scheduler.base_lrs)

def evaluator(name, pairs):
    if type(pairs) != list and type(pairs) == tuple:
        pairs = [pairs]
    n_correct = 0
    n_total = 0
    pk = 0
    k = 30
    qa_eval_list = []
    for output, batch in pairs:
        n_correct += torch.sum((torch.max(output, 1)[1].view(batch.LABEL.size()).data == batch.LABEL.data)).item()
        n_total += batch.LABEL.size(0)
        logit = output.cpu().data.numpy()[:, 1]
        actual = batch.LABEL.cpu().data.numpy()
        qa_eval_list.append((logit, actual))        # Get top k
        # output = (batch, label_size)
        top_k_scores, top_k_indices = torch.topk(output[:,1], k=min(k, output.size(0)), sorted=True)
        top_k_scores_array = top_k_scores.cpu().data.numpy()
        top_k_indices_array = top_k_indices.cpu().data.numpy()
        label = batch.LABEL.cpu().data.numpy()
        tp = 0
        for index in top_k_indices_array:
            if label[index] == 1:
                tp += 1
        pk += tp / k
    if name == "test":
        MAP, MRR, P_30 = evaluation.TWITTER_MAP_MRR(qa_eval_list, pred_fname="pred.test.{}".format(os.getpid()),
                                                    gold_fname="data/qrels.txt",
                                                    id_fname="data/test{}.idb".format(args.train_dataset),
                                                    qid_index=0, docid_index=1, delimiter=' ', model="NN")
        return (n_correct / n_total, P_30, MAP, MRR)
    if name == "valid":
        MAP, MRR, P_30 = evaluation.TWITTER_MAP_MRR(qa_eval_list, pred_fname="pred.valid.{}".format(os.getpid()),
                                                    gold_fname="data/qrels.txt",
                                                    id_fname="data/valid{}.idb".format(args.train_dataset),
                                                    qid_index=0, docid_index=1, delimiter=' ', model="NN")
        return (n_correct / n_total, P_30, MAP)
    if name == "train":
        return (n_correct / n_total, )


# The evaluator output is the input of metrics_comparison
# Used in parameters selection
def metrics_comparison(new_metrics, best_metrics):
    if best_metrics == None or new_metrics[1] >= best_metrics[1]:
        return True
    return False

log = logger.Logger(args.tensorboard)
# The evaluator output is the input of log_printer
def log_printer(name, metrics, loss, epoch=None, iters=None):
    if name == 'train':
        print("{}\tEPOCH:{}\tITER:{}\tACC:{}\tNearest batch training LOSS:{}".format(name, epoch, iters, metrics[0],loss))
        step = int(iters.split('/')[0]) + int(iters.split('/')[1]) * (epoch - 1)
        log.scalar_summary(tag='loss', value=loss, step=step)
    elif name == 'valid':
        print("{}\tACC:{}\tP30:{}MAP:{}\tLOSS:{}".format(name, metrics[0], metrics[1], metrics[2], loss))
        if iters != None and epoch != None and loss != None:
            step = int(iters.split('/')[0]) + int(iters.split('/')[1]) * (epoch - 1)
            log.scalar_summary(tag='valid_loss', value=loss, step=step)
    else:
        print("{}\tACC:{}\tP30:{}\tMAP:{}\tMRR:{}\tLOSS:{}".format(name, metrics[0], metrics[1], metrics[2], metrics[3],loss))
        if iters != None and epoch != None and loss != None:
            step = int(iters.split('/')[0]) + int(iters.split('/')[1]) * (epoch - 1)
            log.scalar_summary(tag='test_loss', value=loss, step=step)


class Trainer(app.TrainAPP):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        self.config.word_num = len(self.QUESTION.vocab)
        self.config.num_classes = len(self.LABEL.vocab)
        # QUESTION and ANSWER use same Field
        stoi, vectors, dim = torch.load(self.config.vector_cache)
        match_embedding = 0
        self.QUESTION.vocab.vectors = torch.Tensor(len(TEXT.vocab), dim)
        for i, token in enumerate(self.QUESTION.vocab.itos):
            wv_index = stoi.get(token, None)
            if wv_index is not None:
                self.QUESTION.vocab.vectors[i] = vectors[wv_index]
                match_embedding += 1
            else:
                self.QUESTION.vocab.vectors[i] = torch.FloatTensor(self.config.word_embed_dim).uniform_(-0.05, 0.05)#normal_(0, 1)
        print("Matching {} out of {}".format(match_embedding, len(self.QUESTION.vocab)))





    def prepare(self, **kwargs):
        super(Trainer, self).prepare(**kwargs)
        self.model.embedding.weight.data.copy_(self.QUESTION.vocab.vectors)
        # print("Start to load tfidf information")
        # tfidf = load_tfidf(stoi=self.QUESTION.vocab.stoi, file_path=self.config.tfidf_file)
        # self.model.tfidf.weight.data.copy_(tfidf)
        # self.model.tfidf.weight.requires_grad = False
        # print("Finish loading tfidf")
        print(self.model)
        print(self.LABEL.vocab.itos)
        print("Training instance : ", len(self.train_iter.dataset))
        print("Valid instance : ", len(self.valid_iter.dataset))
        print("Testing instance : ", len(self.test_iter.dataset))

    def testing(self, epoch):
        with torch.no_grad():
            small_batch_size = 32
            self.model.eval()
            self.test_iter.init_epoch()
            test_result = []
            test_loss = 0
            for test_batch_idx, test_batch in enumerate(self.test_iter):
                small_batch = (test_batch.QUESTION.size(0) - 1) // small_batch_size + 1
                logit = []
                for i in range(small_batch):
                    if i == small_batch - 1:
                        sent1 = test_batch.QUESTION[small_batch_size * i:]
                        sent2 = test_batch.ANSWER[small_batch_size * i:]
                        label = test_batch.LABEL[small_batch_size * i:]
                        ext = test_batch.IRFEATURE[small_batch_size * i:]
                    else:
                        sent1 = test_batch.QUESTION[small_batch_size * i:small_batch_size * (i + 1)]
                        sent2 = test_batch.ANSWER[small_batch_size * i:small_batch_size * (i + 1)]
                        label = test_batch.LABEL[small_batch_size * i: small_batch_size * (i + 1)]
                        ext = test_batch.IRFEATURE[small_batch_size * i: small_batch_size * (i + 1)]
                    if self.config.ext_feats:
                        test_output_ = self.model(sent1, sent2, ext)
                    else:
                        test_output_ = self.model(sent1, sent2, None)
                    logit.append(test_output_[0])
                    test_loss += self.criterion(test_output_, label).item()
                test_output = torch.cat(logit, dim=0)
                test_result.append((test_output, test_batch))
            test_metrics = self.evaluator("test", test_result)
            self.log_printer("test", loss=test_loss, metrics=test_metrics)

    def train(self):
        epoch = 0
        iterations = 0
        best_metrics = None
        iters_not_improved = 0
        small_batch_size = args.small_batch_size
        time_output = open("training_time_{}".format(args.model_type), "w")
        one_epoch_flag = False
        true_batch_counter = 0
        while True:
            epoch += 1
            if epoch > 15:
                print("Stopping")
                break
            self.train_iter.init_epoch()
            self.optimizer.schedule()
            for batch_idx, batch in enumerate(self.train_iter):
                if not one_epoch_flag:
                    true_batch_counter += 1
                iterations += 1
                self.model.train()
                train_loss = 0
                small_batch = (batch.QUESTION.size(0) - 1) // small_batch_size + 1
                logit = []
                start_training_time = time.time()
                for i in range(small_batch):
                    self.optimizer.zero_grad()
                    if i == small_batch - 1:
                        sent1 = batch.QUESTION[small_batch_size * i:]
                        sent2 = batch.ANSWER[small_batch_size * i:]
                        label = batch.LABEL[small_batch_size * i:]
                        ext = batch.IRFEATURE[small_batch_size * i:]
                    else:
                        sent1 = batch.QUESTION[small_batch_size * i:small_batch_size * (i + 1)]
                        sent2 = batch.ANSWER[small_batch_size * i:small_batch_size * (i + 1)]
                        label = batch.LABEL[small_batch_size * i:small_batch_size * (i + 1)]
                        ext = batch.IRFEATURE[small_batch_size * i: small_batch_size * (i + 1)]
                    if self.config.ext_feats:
                        output_ = self.model(sent1, sent2, ext)
                    else:
                        output_ = self.model(sent1, sent2, None)
                    logit.append(output_[0])
                    loss = self.criterion(output_, label)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()

                end_training_time = time.time()
                batch_size = batch.QUESTION.size(0)
                elapsed = end_training_time - start_training_time
                averaged_elpased = elapsed / batch_size
                time_output.write("{}\t{}\t{}\t{}\t{}\n".format(start_training_time,
                                                            end_training_time,
                                                            elapsed,
                                                            averaged_elpased,
                                                            batch_size))
                time_output.flush()
                output = torch.cat(logit, dim=0)
                # We generate metrics for each batch, not all batches so far
                metrics = self.evaluator("train", (output, batch))


                with torch.no_grad():
                    if iterations % self.args.valid_every == 1:
                        self.model.eval()
                        self.valid_iter.init_epoch()
                        valid_result = []
                        valid_loss = 0
                        for valid_batch_idx, valid_batch in enumerate(self.valid_iter):
                            small_batch = (valid_batch.QUESTION.size(0) - 1) // small_batch_size + 1
                            logit = []
                            for i in range(small_batch):
                                if i == small_batch - 1:
                                    sent1 = valid_batch.QUESTION[small_batch_size * i:]
                                    sent2 = valid_batch.ANSWER[small_batch_size * i:]
                                    label = valid_batch.LABEL[small_batch_size * i :]
                                    ext = valid_batch.IRFEATURE[small_batch_size * i :]
                                else:
                                    sent1 = valid_batch.QUESTION[small_batch_size * i:small_batch_size * (i + 1)]
                                    sent2 = valid_batch.ANSWER[small_batch_size * i:small_batch_size * (i + 1)]
                                    label = valid_batch.LABEL[small_batch_size * i: small_batch_size * (i + 1)]
                                    ext = valid_batch.IRFEATURE[small_batch_size * i: small_batch_size * (i + 1)]
                                if self.config.ext_feats:
                                    valid_output_ = self.model(sent1, sent2, ext)
                                else:
                                    valid_output_ = self.model(sent1, sent2, None)
                                logit.append(valid_output_[0])
                                valid_loss += self.criterion(valid_output_, label).item()
                            valid_output = torch.cat(logit, dim=0)
                            valid_result.append((valid_output, valid_batch))
                        valid_metrics = self.evaluator("valid", valid_result)
                        self.log_printer("valid", loss=valid_loss, metrics=valid_metrics)

                        if self.metrics_comparison(valid_metrics, best_metrics):
                            best_metrics = valid_metrics
                            torch.save(self.model, self.snapshot_path)
                            print("Saving model to {}".format(self.snapshot_path))
                            self.testing(epoch)

                if iterations % self.args.log_every == 0:
                    self.log_printer("train", loss=train_loss, metrics=metrics, epoch= epoch, iters= "{}/{}".format(batch_idx ,true_batch_counter if one_epoch_flag else -1))
            one_epoch_flag = True

def load_tfidf(stoi, file_path):
    word_weights = json.load(open(file_path))
    tfidf = torch.Tensor(len(stoi), 1)
    for word in stoi:
        idx = stoi[word]
        if word in word_weights:
            tfidf[idx] = word_weights[word]
        else:
            tfidf[idx] = 1
    return tfidf

if __name__=='__main__':

    trainer = Trainer(args=args, fields=fields, include_test=include_test,
                            batch_size_fn_train=batch_size_fn_zoo['train{}'.format(args.train_dataset)],
                            batch_size_fn_valid=batch_size_fn_zoo['valid{}'.format(args.train_dataset)],
                            batch_size_fn_test=batch_size_fn_zoo['test{}'.format(args.train_dataset)],
                            train_shuffle=args.shuffle)
    if args.model_type == 'attn':
        model = Attention
    elif args.model_type == "qac":
        model = FastDynamic
    elif args.model_type == "baselines":
        model = SM
    elif args.model_type == "attn_dot":
        model = AttentionDot
    else:
        print("Wrong Model Type")
        exit()
    trainer.prepare(model=model, optimizer=optimizer, criterion=criterion(), evaluator=evaluator,
                           metrics_comparison=metrics_comparison, log_printer=log_printer)
    trainer.train()
