#!/usr/bin/env python3

import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys


def parse_arguments():
    class Parser(argparse.ArgumentParser):
        pass
    parser = Parser(
        description="package all release file.",
        usage="""
        Default behavior is --build_rpm --build_deb  --build_dir path --config Debug
        """,
        fromfile_prefix_chars="@",
    )
    # Main arguments
    parser.add_argument("--build_rpm", action="store_true", default=False)
    parser.add_argument("--build_deb", action="store_true", default=False)
    parser.add_argument("--build_dir", required=True, help="Path to the build directory.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["Debug"],
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration(s) to build.",
    )
    parser.add_argument(
        "--use_maca", action='store_true', help="Build with MACA SDK.")
    parser.add_argument(
        "--package_dir",
        type=str,
        default="",
        help="Specify packaging directory.",
    )
    args, unknown = parser.parse_known_args()
    return args

def get_config_build_dir(build_dir, config):
    # build directory per configuration
    return os.path.join(build_dir, config)


def package(
    source_dir,
    build_rpm,
    build_deb,
    build_dir,
    use_maca,
    configs,
    package_dir
):
    project_dir = os.path.join(source_dir, '..', '..')
    for config in configs:
        config_build_dir = get_config_build_dir(build_dir, config)

        arch = platform.machine()
        #mkdir
        if package_dir != "":
            print(package_dir)
            onnxruntime_dir = os.path.join(package_dir, "onnxruntime-maca")
        else:
            onnxruntime_dir = os.path.join(config_build_dir, "onnxruntime-maca")
        print("Package dir: ", onnxruntime_dir)

        #clean package_dir
        if os.path.exists(onnxruntime_dir):
            print("clean package_dir...")
            shutil.rmtree(onnxruntime_dir)

        dirs = []

        #package 1 level dir
        python_package_dir = os.path.join(onnxruntime_dir, "python")
        dirs.append(python_package_dir)
        include_dir = os.path.join(onnxruntime_dir, "include")
        dirs.append(include_dir)
        lib_dir = os.path.join(onnxruntime_dir, "lib")
        dirs.append(lib_dir)
        bin_dir = os.path.join(onnxruntime_dir, "bin")
        dirs.append(bin_dir)
        sample_dir = os.path.join(onnxruntime_dir, "sample")
        dirs.append(sample_dir)

        #python
        tools_package_dir = os.path.join(python_package_dir, "tools")
        dirs.append(tools_package_dir)

        #sample
        sample_cpp_dir = os.path.join(sample_dir, "c++")
        dirs.append(sample_cpp_dir)
        sample_python_dir = os.path.join(sample_dir, "python")
        dirs.append(sample_python_dir)


        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)


        #Copy wheel
        dist_dir = os.path.join(config_build_dir, 'dist')
        onnxruntime_whl = glob.glob(os.path.join(dist_dir, 'onnxruntime_*.whl'))
        tools_whl = glob.glob(os.path.join(dist_dir, 'maca_*.whl'))

        for file_path in onnxruntime_whl:
            shutil.copy(file_path, python_package_dir)
        for file_path in tools_whl:
            shutil.copy(file_path, tools_package_dir)

        #Copy include
        shutil.copy(os.path.join(project_dir,"include/onnxruntime/core/session/onnxruntime_c_api.h"), include_dir)
        shutil.copy(os.path.join(project_dir,"include/onnxruntime/core/session/onnxruntime_cxx_api.h"), include_dir)
        shutil.copy(os.path.join(project_dir,"include/onnxruntime/core/session/onnxruntime_cxx_inline.h"), include_dir)
        shutil.copy(os.path.join(project_dir,"include/onnxruntime/core/providers/cpu/cpu_provider_factory.h"), include_dir)
        shutil.copy(os.path.join(project_dir,"include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h"), include_dir)
        shutil.copy(os.path.join(project_dir,"include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h"), include_dir)
        shutil.copy(os.path.join(project_dir,"include/onnxruntime/core/framework/provider_options.h"), include_dir)
        shutil.copy(os.path.join(project_dir,"include/onnxruntime/core/graph/constants.h"), include_dir)

        #Copy lib
        onnxruntime_so = glob.glob(os.path.join(config_build_dir,'libonnxruntime.so*'))
        for file_path in onnxruntime_so:
            if os.path.islink(file_path):
                if not os.path.exists(os.path.join(lib_dir, 'libonnxruntime.so')):
                    shutil.copy(file_path, lib_dir, follow_symlinks=False)
            else:
                shutil.copy(file_path, lib_dir)

        #Copy bin
        shutil.copy(os.path.join(config_build_dir, "maca_test"), bin_dir)

        #Copy Sample
        shutil.copytree(os.path.join(project_dir, "cmake/external/onnx/onnx/backend/test/data/node/test_add"), os.path.join(sample_dir, "test_add"))
        shutil.copytree(os.path.join(project_dir, "onnxruntime/test/providers/maca/common"), os.path.join(sample_cpp_dir, "common"))
        shutil.copytree(os.path.join(project_dir, "onnxruntime/test/providers/maca/custom_op_test"), os.path.join(onnxruntime_dir, "custom_op_test"))
        shutil.copy(os.path.join(project_dir, "onnxruntime/test/providers/maca/maca_test.cc"), sample_cpp_dir)
        shutil.copy(os.path.join(project_dir, "onnxruntime/test/providers/maca/CMakeLists.txt"), sample_cpp_dir)
        shutil.copy(os.path.join(project_dir, "onnxruntime/test/providers/maca/ReadMe.md"), sample_cpp_dir)
        shutil.copy(os.path.join(project_dir, "onnxruntime/test/python/maca_test.py"), sample_python_dir)


        if build_deb:
            if arch == 'x86_64':
                arch = 'amd64'
            #debian
            onnxruntime_deb_dir = os.path.join(config_build_dir, "onnxruntime-maca_deb")
            print("Deb_dir: ", onnxruntime_deb_dir)

            #clean deb_dir
            if os.path.exists(onnxruntime_deb_dir):
                print("clean onnxruntime-maca_deb dir...")
                shutil.rmtree(onnxruntime_deb_dir)

            maca_dir = os.path.join(onnxruntime_deb_dir, "opt/maca-ai")
            if not os.path.exists(maca_dir):
                os.makedirs(maca_dir)

            shutil.copytree(onnxruntime_dir, os.path.join(maca_dir, "onnxruntime-maca"), symlinks=True)

            debian_dir = os.path.join(onnxruntime_deb_dir, "DEBIAN")
            if not os.path.exists(debian_dir):
                os.makedirs(debian_dir)

            shutil.copy(os.path.join(project_dir, "package/deb/postinst"), debian_dir)
            shutil.copy(os.path.join(project_dir, "package/deb/prerm"), debian_dir)
            control_file = os.path.join(debian_dir, "control")

            package_name = "onnxruntime-maca"
            ort_version_number = ""
            with open("VERSION_NUMBER") as f:
                ort_version_number = f.readline().strip()
            maca_ai_version = os.getenv('MACA_AI_VERSION')
            if not maca_ai_version:
                maca_ai_version = "0.0.0.0"

            with open(control_file, 'w') as f:
                f.write("Package: " + package_name + "\n")
                f.write("Depends: \n")
                f.write("Version: " + maca_ai_version + "\n")
                f.write("Architecture: " + arch + "\n")
                f.write("Maintainer: METAX\n")
                f.write("Description: onnxruntime-maca deb package\n")

            if package_dir != "":
                deb_dir = os.path.join(package_dir, "ai_deb")
            else:
                deb_dir = os.path.join(config_build_dir, "ai_deb")
            if not os.path.exists(deb_dir):
                os.makedirs(deb_dir)

            deb_file = os.path.join(deb_dir, package_name+"_"+ort_version_number+"-"+maca_ai_version+"_"+arch+".deb")
            print("deb_file_path: ", deb_file)
            err = os.system("fakeroot dpkg-deb -b " + onnxruntime_deb_dir + " " + deb_file)
            if err:
                exit(-1)

        if build_rpm:
            if arch == 'amd64':
                arch = 'x86_64'
            #rpm
            onnxruntime_rpm_topbuild = os.path.join(config_build_dir, "onnxruntime-maca_rpm_build")
            onnxruntime_rpm_dir = os.path.join(config_build_dir, "onnxruntime-maca_rpm")

            print("rpm_dir: ", onnxruntime_rpm_dir)

            #clean rpm_dir
            if os.path.exists(onnxruntime_rpm_dir):
                print("clean onnxruntime-maca_rpm dir...")
                shutil.rmtree(onnxruntime_rpm_dir)

            maca_dir = os.path.join(onnxruntime_rpm_dir, "opt/maca-ai")
            if not os.path.exists(maca_dir):
                os.makedirs(maca_dir)

            shutil.copytree(onnxruntime_dir, os.path.join(maca_dir, "onnxruntime-maca"), symlinks=True)

            rpm_dir = os.path.join(onnxruntime_rpm_dir, "RPM")
            if not os.path.exists(rpm_dir):
                os.makedirs(rpm_dir)


            package_name = "onnxruntime-maca"
            ort_version_number = ""
            with open("VERSION_NUMBER") as f:
                ort_version_number = f.readline().strip()
            maca_ai_version = os.getenv('MACA_AI_VERSION')
            if not maca_ai_version:
                maca_ai_version = "0.0.0.0-00"
            versions = maca_ai_version.split('-', 1)
            if len(versions) == 2:
                [rpm_version, rpm_release] = versions
            else:
                [rpm_version, rpm_release] = maca_ai_version.rsplit(".",1)

            if package_dir != "":
                rpm_dir = os.path.join(package_dir, "ai_rpm")
            else:
                rpm_dir = os.path.join(config_build_dir, "ai_rpm")
            if not os.path.exists(rpm_dir):
                os.makedirs(rpm_dir)

            sub_dirs = ['BUILD', 'BUILDROOT', "RPMS", 'SOURCES', 'SPECS/SRPMS']
            for dir_path in sub_dirs:
                full_path = os.path.join(onnxruntime_rpm_topbuild, dir_path)
                if not os.path.exists(full_path):
                    os.makedirs(full_path)
            spec_file = os.path.join(onnxruntime_rpm_topbuild,"SPECS/onnxruntime-maca.spec")
            rpm_file = os.path.join(rpm_dir, package_name+"_"+ort_version_number+"-"+maca_ai_version+"_"+arch+".rpm")
            print(rpm_file)
            print(onnxruntime_rpm_dir)

            with open(spec_file,'w') as fs:
                fs.write("%define  _build_id_links none" + "\n")
                fs.write("%define _topdir " + onnxruntime_rpm_topbuild + "\n")
                fs.write("Name:     " +package_name + "\n")
                fs.write("Version:  " + rpm_version + "\n")
                fs.write("Release:  " + rpm_release + "\n")
                fs.write("Summary:  onnxruntime libs, tools, python files" + "\n")
                fs.write("BuildArch:" + arch + "\n")
                fs.write("License:  GPL" + "\n")
                fs.write("%description" + "\n")
                fs.write("onnxruntime libs, tools,sample,python files" + "\n")
                fs.write("%prep" + "\n")
                fs.write("mkdir -p $RPM_BUILD_ROOT" + "\n")
                fs.write("%build" + "\n")
                fs.write("%install" + "\n")
                fs.write("%{__cp} -r \\" + onnxruntime_rpm_dir+"/*" +" $RPM_BUILD_ROOT" + "\n")
                fs.write("%clean" + "\n")
                fs.write("rm -rf $RPM_BUILD_ROOT" + "\n")
                fs.write("%files" + "\n")
                fs.write("/opt/maca-ai/onnxruntime-maca"+ "\n")
                fs.write("%pre" + "\n")
                fs.write("%post" + "\n")
                fs.write("echo -e /opt/maca-ai/onnxruntime-maca/lib > /etc/ld.so.conf.d/libonnxruntime-maca.conf && ldconfig" + "\n")
                fs.write("echo -e \"onnxruntime-maca package installed successfull\"" + "\n")
                fs.write("%changelog")
            err = os.system("rpmbuild -D \"_topdir " + onnxruntime_rpm_topbuild +"\" -bb " + spec_file)
            if err:
                exit(-1)
            os.system("cp " + onnxruntime_rpm_topbuild + "/RPMS/"+ arch +"/* " + rpm_file)
            shutil.rmtree(onnxruntime_rpm_topbuild)

def main():
    args = parse_arguments()

    configs = set(args.config)
    build_rpm = args.build_rpm
    build_deb = args.build_deb
    build_dir = args.build_dir
    script_dir = os.path.realpath(os.path.dirname(__file__))
    package_dir = args.package_dir
    use_maca = args.use_maca

    package(
        script_dir,
        build_rpm,
        build_deb,
        build_dir,
        use_maca,
        configs,
        package_dir
    )

if __name__ == "__main__":
    sys.exit(main())
