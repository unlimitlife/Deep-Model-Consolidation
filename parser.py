import argparse
from datetime import datetime
from config import method_config, data_config


def _to_string(args, parser):
    ret = "main.py"
    for action in parser._get_positional_actions():
        ret += " " + str(getattr(args, action.dest))
    for action in parser._get_optional_actions():
        if action.default != "==SUPPRESS==":
            ret += " " + action.option_strings[0] + " " + str(getattr(args, action.dest))
    return ret


parser = argparse.ArgumentParser(description='python3 main.py "model" "dataset" "curriculum" "method" --option')

# positional arguments
parser.add_argument("model", type=str, choices=['convnet','resnet18', 'resnet32', 'wide_resnet'],
                    help='specifies model')
parser.add_argument("dataset", type=str, choices=list(data_config.keys()),
                    help='specifies dataset')
parser.add_argument("curriculum", type=str,
                    choices=['basic', 'rand1', 'rand2', 'base1', 'base2', 'best', 'worst', 'super', 'super1', 'super2',
                             'another0', 'another1', 'another2', 'another3', 'another4', 'another5', 'another6',
                             'another7', 'another8', 'another9',
                             'best1', 'best2', 'best3', 'best4', 'best5'],
                    help='specifies curriculum setting in config.py')
parser.add_argument("method", type=str,
                    choices=list(method_config.keys()),
                    help='specifies training method')

# optional arguments
parser.add_argument("--desc", type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'),
                    help='training description [default: cur_time]')
parser.add_argument("--classes-per-task", type=int, default=10,
                    help='number of classes per task [default: 10]')
parser.add_argument("--memory-cap", type=int, default=2000,
                    help='Sample memory capacity [default: 2000]')

args = parser.parse_args()
command = _to_string(args, parser)
