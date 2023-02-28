import os
from multitask_classifier import test_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--use_gpu", action='store_true')

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    
    # training setting
    parser.add_argument("--output_dir", type=str, help="dir for saved model (.pt) and prediction files (.csv)",
                        default="result/tmp")

    args = parser.parse_args()
    
    args.sst_dev_out = os.path.join(args.output_dir, "sst-dev-output.csv")
    args.sst_test_out = os.path.join(args.output_dir, "sst-test-output.csv")
    args.para_dev_out = os.path.join(args.output_dir, "para-dev-output.csv")
    args.para_test_out = os.path.join(args.output_dir, "para-test-output.csv")
    args.sts_dev_out = os.path.join(args.output_dir, "sts-dev-output.csv")
    args.sts_test_out = os.path.join(args.output_dir, "sts-test-output.csv")
    
    return args

if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test_model(args)
