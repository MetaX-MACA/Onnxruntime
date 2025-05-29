# coding=utf-8
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import time
from maca_quantizer.maca_quantize_runner import MacaQuantizeRunner
from maca_quantizer.utils.utils import maca_info

def main_parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--config', '-c', required=True, help='MacaQuantize config file')
    parser.add_argument('--workfolder', '-w', default=None, help='MacaQuantize run workspaceFolder')
    parser.add_argument('--mode', '-m', type=str, default='auto', choices=['debug','fast', 'naive','auto'],  help='MacaQuantize run mode')
    parser.add_argument('--version', '-v', action='store_true', default=False, help='MacaQuantize Version')
    arg = parser.parse_args()
    return arg

if __name__=="__main__":
    args = main_parse_args()
    
    if args.workfolder is not None:
        work_folder = os.path.abspath(args.workfolder)
    else:
        work_folder = os.path.abspath(os.path.dirname(args.config) + "../../../")
    maca_info("Workspace Folder: {}".format(work_folder))
    os.chdir(work_folder)

    maca_info("Current workspace Folder: {}".format(os.getcwd()))
    maca_info("MacaQuantizer Version: {}".format(MacaQuantizeRunner.version()))
    maca_info("MacaQuantizer Device:  {}".format(MacaQuantizeRunner.check_device()))

    start_time = time.time()
    obj = MacaQuantizeRunner(args.config, mode=args.mode)
    quant_graph = obj.run()
    end_time = time.time()
    maca_info("cost time: {:2f} second".format(end_time - start_time))