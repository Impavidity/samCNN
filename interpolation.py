import argparse
import math
import subprocess

def read_pred_file(file_name, args):
    fin = open(file_name)
    result_list = []
    qid_list = []
    docid_list = []
    pair_list = []
    for line in fin:
        items = line.strip().split()
        qid = int(items[0])
        docid = int(items[2])
        score = math.exp(float(items[4])) if args.model_use_exp else float(items[4])
        pair_list.append((qid, docid, score))
    pair_list = sorted(pair_list, key=lambda x: (x[0], x[1]))
    for item in pair_list:
        result_list.append(item[2])
        qid_list.append(item[0])
        docid_list.append(item[1])
    return qid_list, docid_list, result_list

def evaluation(baseline_result_list, model_result_list, qid_list, docid_list, alpha, pred_file_name):
    gold_fname = "data/qrels.txt"
    fout = open(pred_file_name, "w")
    for qid, docid, baseline_score, model_score in zip(qid_list, docid_list, baseline_result_list, model_result_list):
        score = model_score + alpha * math.log(baseline_score)
        fout.write("{} Q0 {} 0 {} Inter\n".format(qid, docid, score))
    fout.flush()
    fout.close()
    trec_eval_path = 'trec_eval-9.0.5/trec_eval'
    trec_out = subprocess.check_output([trec_eval_path, gold_fname, pred_file_name])
    trec_out_lines = str(trec_out, 'utf-8').split('\n')
    mean_average_precision = float(trec_out_lines[5].split('\t')[-1])
    mean_reciprocal_rank = float(trec_out_lines[9].split('\t')[-1])
    p_30 = float(trec_out_lines[25].split('\t')[-1])
    return p_30, mean_average_precision

def parameter_selection(baseline_result_list, model_result_list, qid_list, docid_list, args):
    best_p30_alpha = 0
    best_map_alpha = 0
    best_map = 0
    best_p30 = 0
    print("ALPHA\tP30\tMAP")
    for i in range(2001):
        alpha = i / 10000
        p30, map = evaluation(baseline_result_list, model_result_list, qid_list, docid_list, alpha, "temp{}".format(args.train_dataset))
        print("{}\t{}\t{}".format(alpha, p30, map))
        if p30 > best_p30:
            best_p30_alpha = alpha
            best_p30 = p30
        if map > best_map:
            best_map_alpha = alpha
            best_map = map
    return best_p30_alpha, best_map_alpha

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--baseline_data", type=str)
    argparser.add_argument("--model_data", type=str)
    argparser.add_argument("--baseline_eval", type=str)
    argparser.add_argument("--model_eval", type=str)
    argparser.add_argument("--model_use_exp", action="store_true", default=False)
    argparser.add_argument("--train_dataset", type=str)
    args = argparser.parse_args()
    baseline_qid_list, baseline_docid_list, baseline_result_list = read_pred_file(args.baseline_data, args)
    model_qid_list, model_docid_list, model_result_list = read_pred_file(args.model_data, args)
    assert(model_qid_list == baseline_qid_list)
    assert(model_docid_list == baseline_docid_list)
    best_p30_alpha, best_map_alpha = parameter_selection(baseline_result_list, model_result_list, model_qid_list, model_docid_list, args)
    print("Best P30 alpha:", best_p30_alpha, "Best map alpha:", best_map_alpha)
    eval_baseline_qid_list, eval_baseline_docid_list, eval_baseline_result_list = read_pred_file(args.baseline_eval, args)
    eval_model_qid_list, eval_model_docid_list, eval_model_result_list = read_pred_file(args.model_eval, args)
    #assert (eval_model_qid_list == eval_baseline_qid_list)
    #assert (eval_model_docid_list == eval_baseline_docid_list)
    p_30_eval, map_eval = evaluation(eval_baseline_result_list, eval_model_result_list, eval_model_qid_list, eval_model_docid_list, best_map_alpha, "interpolation/{}".format(args.train_dataset))
    print("result:", p_30_eval, map_eval)
