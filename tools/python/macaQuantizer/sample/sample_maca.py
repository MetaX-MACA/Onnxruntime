# coding=utf-8
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from maca_quantizer.maca_quantize_runner import MacaQuantizeRunner
from maca_quantizer.utils.utils import maca_info

def main_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default=None, help='MacaQuantize config file')
    parser.add_argument('--mode', '-m', type=str, default='auto', choices=['debug','fast','naive','auto'],  help='MacaQuantize run mode')
    parser.add_argument('--ep', '-e', type=str, default='mxcuda', choices=['mxcuda','mxppl'],  help='MacaQuantize run mode')
    parser.add_argument('--version', '-v', action='store_true', default=False, help='MacaQuantize Version')
    parser.add_argument('--random', '-r', action='store_true', default=False, help='Random data to quantize')
    arg = parser.parse_args()
    return arg

if __name__=="__main__":
    start_time = time.time()
    args = main_parse_args()
    if args.version and args.config is None:
        maca_info("MacaQuantizer Version: {}".format(MacaQuantizeRunner.version()))
        maca_info("MacaQuantizer Device:  {}".format(MacaQuantizeRunner.check_device()))
        exit(0)
    if args.config is not None:
        maca_info("MacaQuantizer Version: {}".format(MacaQuantizeRunner.version()))
        maca_info("MacaQuantizer Device:  {}".format(MacaQuantizeRunner.check_device()))
        obj = MacaQuantizeRunner(args.config, ep=args.ep ,mode=args.mode, random=args.random)
        quant_graph = obj.run()
    maca_info(f"Quantization Cost Time: {(time.time() - start_time) / 60} min")