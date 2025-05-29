#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import shutil

def parse_arguments():
    class Parser(argparse.ArgumentParser):
        pass
    parser = Parser(
        description="macaQuantizer Tool wheel build.",
        usage="""
        Default behavior is  --build_dir path --config Debug
        """,
        fromfile_prefix_chars="@",
    )
    # Main arguments
    parser.add_argument("--build_dir", required=True, help="Path to the build directory.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["Debug"],
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration(s) to build.",
    )

    args, unknown = parser.parse_known_args()
    return args

def get_config_build_dir(build_dir, config):
    # build directory per configuration
    return os.path.join(build_dir, config)


def build_python_wheel(
    source_dir,
    build_dir,
    configs,
):
    
    for config in configs:
        config_build_dir = os.path.join(get_config_build_dir(build_dir, config), "dist")
        args = [sys.executable, os.path.join(source_dir, "setup.py"), "bdist_wheel", "--dist-dir={}".format(config_build_dir)]
        out = subprocess.run(args, cwd=source_dir)
        if out.returncode:
            exit(-1)

def main():
    args = parse_arguments()

    configs = set(args.config)
    build_dir = args.build_dir
    script_dir = os.path.realpath(os.path.dirname(__file__))

    tools = ['macaConverter',  "macaQuantizer", "macaPrecision"]
    modules_name = ['maca_converter',  'maca_quantizer', 'maca_precision']
    for tool, module_name  in zip(tools, modules_name):
        source_dir = os.path.join(script_dir, tool)
        build_python_wheel(
            source_dir,
            build_dir,
            configs
        )

        temp_build = os.path.join(source_dir, "build")
        if os.path.exists(temp_build):
            shutil.rmtree(temp_build)
        
        temp_egg = os.path.join(source_dir, "{}.egg-info".format(module_name))
        if os.path.exists(temp_egg):
            shutil.rmtree(temp_egg)

if __name__ == "__main__":
    sys.exit(main())
 
