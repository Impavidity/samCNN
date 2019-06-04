import subprocess

def TWITTER_MAP_MRR(pairs, pred_fname="pred.txt", gold_fname=None, id_fname=None,
                    qid_index=None, docid_index=None, delimiter=' ', model="model"):
    if id_fname == None or gold_fname == None or qid_index == None or docid_index == None:
        print("You need to pass filename of qrel or qid/docid to the function")
        exit()
    qid_file = open(id_fname)
    id_list = []
    for line in qid_file.readlines():
        line = line.strip().split(delimiter)
        qid = line[qid_index]
        docid = line[docid_index]
        id_list.append((qid, docid))
    results_file = open(pred_fname, "w")
    results_template = "{qid} Q0 {docno} 0 {sim} {model}\n"
    counter = 0
    for pair in pairs:
        for predicted in pair[0]:
            qid = id_list[counter][0]
            docid = id_list[counter][1]
            results_file.write(results_template.format(qid=qid, docno=docid, sim=predicted, model=model))
            counter += 1
    if counter != len(id_list):
        print("Counter is not equal the total number of the documents")
        exit()
    results_file.flush()
    results_file.close()
    trec_eval_path = 'trec_eval-9.0.5/trec_eval'
    trec_out = subprocess.check_output([trec_eval_path, gold_fname, pred_fname])
    trec_out_lines = str(trec_out, 'utf-8').split('\n')
    mean_average_precision = float(trec_out_lines[5].split('\t')[-1])
    mean_reciprocal_rank = float(trec_out_lines[9].split('\t')[-1])
    p_30 = float(trec_out_lines[25].split('\t')[-1])

    #os.remove(pred_fname)

    return mean_average_precision, mean_reciprocal_rank, p_30