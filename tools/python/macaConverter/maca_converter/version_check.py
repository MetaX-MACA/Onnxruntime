from packaging.version import Version, parse
import numpy

np_default_version = Version('1.19.5')

def check(sw, version, target=''):
    if target=='' and sw == 'tensorflow':
        np_version = numpy.__version__
        np_version = parse(np_version)

        if np_version <= np_default_version:
            tf_target = Version('2.4.0')
            tf_version = parse(version)
            if tf_version != tf_target:
                print('####################################################')
                print('WARNING: your tensowflow version is', version)
                print('suggested version is 2.4.0')
                print('####################################################')
        elif np_version > np_default_version:  
            tf_target_min = Version('2.7.0')
            tf_target_max= Version('2.7.4')
            tf_version = parse(version)
            if tf_version < tf_target_min or tf_version > tf_target_max:
                print('####################################################')
                print('WARNING: your tensowflow version is', version)
                print('suggested version is between 2.7.0 and 2.7.4')
                print('####################################################')
    elif target != '':
        sw_target = Version(target)
        sw_version = parse(version)
        if sw_version != sw_target:
            print('####################################################')
            print('WARNING: your', sw, 'version is', version)
            print('suggested version is', target)
            print('####################################################')



