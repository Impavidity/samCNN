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

command="python -u prediction.py --train_dataset {} --model_type {} --gating_source {} --gpu {} {}".format(args.train_dataset,
                                                                                                   args.model_type,
                                                                                                   args.gating_source,
                                                                                                   args.gpu,
                                                                                                   feature_tag)

train_log_dir = "{}_{}_{}_{}log".format(args.model_type, args.gating_source, args.train_dataset, "feature_" if args.feature else "")
model_dir="{}_{}_{}_{}model".format(args.model_type, args.gating_source, args.train_dataset, "feature_" if args.feature else "")
tester = script.RandomTester(log_dir="eval_{}_{}_{}_{}log".format(args.model_type, args.gating_source, args.train_dataset, "feature_" if args.feature else ""))
tester.add_pipeline(script=command, model_arg="--trained_model", result_parser=None, log_dir=None, model_dir=model_dir, ignore_last=False)
tester.start()