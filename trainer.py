import script
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--train_dataset", default="123", type=str)
argparser.add_argument("--model_type", default="attn", type=str)
argparser.add_argument("--gating_source", default="ave", type=str)
argparser.add_argument("--gpu", default="0", type=str)
argparser.add_argument("--thread", default=3, type=int)
argparser.add_argument("--feature", default=False, action="store_true")
args = argparser.parse_args()

feature_tag = "--ext_feats" if args.feature else ""


trainer_attn = script.RandomTrainer(
    script="python -u main.py --train_dataset {} --model_type {} --gating_source {} --gpu {} {}".format(args.train_dataset,
                                                                                                   args.model_type,
                                                                                                   args.gating_source,
                                                                                                   args.gpu,
                                                                                                   feature_tag),
    random_seed_arg="--seed",
    model_prefix_arg="--prefix",
    save_path_arg="--save_path",
    log_dir="{}_{}_{}_{}log".format(args.model_type, args.gating_source, args.train_dataset, "feature_" if args.feature else ""),
    model_dir="{}_{}_{}_{}model".format(args.model_type, args.gating_source, args.train_dataset, "feature_" if args.feature else ""),
    round_num=10
)
trainer_attn.start(process_num=args.thread)