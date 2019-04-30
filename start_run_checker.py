#!/usr/bin/env python
import os, glob, sys, datetime, subprocess, re
from time import sleep

LAST_PROCESSED_RUN = 0 #just for initialization
TIME_BETWEEN_CHECKS = 60 #time in s between checking if a new event file has been written
STATUS_FILE = '/SNS/MANDI/shared/autoreduce/autoreduce_status.txt'

def read_last_processed_run(fname='/SNS/MANDI/shared/autoreduce/last_run_processed.dat'):
    with open(fname,'r') as f:
        l = f.readline()
    return int(l)

def write_last_processed_run(runNumber, fname='/SNS/MANDI/shared/autoreduce/last_run_processed.dat'):
    with open(fname,'w') as f:
        l = f.write(str(runNumber))


#Initialization
try:
    LAST_PROCESSED_RUN = read_last_processed_run()
except:
    raise UserWarning("Cannot read last_run_processed.dat to know which run to process")
NEXT_RUN_TO_PROCESS = LAST_PROCESSED_RUN + 1

with open(STATUS_FILE,'r') as f:
    _ = f.readline()
    ipts_number = int(f.readline())
while True:

    LAST_PROCESSED_RUN = read_last_processed_run()
    NEXT_RUN_TO_PROCESS = LAST_PROCESSED_RUN + 1
    print '{2}: Last processed run is {0}, looking for run {1}'.format(LAST_PROCESSED_RUN, NEXT_RUN_TO_PROCESS, str(datetime.datetime.now()))
    nxs_filename = '/SNS/MANDI/IPTS-{0}/nexus/MANDI_{1}.nxs.h5'.format(ipts_number, NEXT_RUN_TO_PROCESS)
    try:
        #print 'Trying to find file {}'.format(nxs_filename)
        #p = subprocess.check_output(["findnexus", "-i", "MANDI", "--event", "{0}".format(NEXT_RUN_TO_PROCESS)])
        file_exists = os.path.isfile(nxs_filename)
        if not file_exists:
            sleep(TIME_BETWEEN_CHECKS)
        else: #we have the file
            outputdir = '/SNS/MANDI/IPTS-{0}/shared/autoreduce/'.format(ipts_number)
            python_command = '/SNS/users/ntv/workspace/mantid/release/bin/mantidpython'
            script_name = '/SNS/MANDI/shared/autoreduce/mandi_singlerun.py'
            mtz_script = '/SNS/MANDI/shared/autoreduce/mandi_createmtz.py'
            config_filename = '/SNS/MANDI/IPTS-{0}/shared/autoreduce/mandi_autoreduce.config'.format(ipts_number)
            output_dir = outputdir
            run_number = NEXT_RUN_TO_PROCESS
            config_file_list = glob.glob(config_filename)
            if len(config_file_list) != 1:
                raise UserWarning("Config file {0} does not exist.  Cannot do profile fitting.".format(config_filename))
            else: #We have a config file we can use
                command_pf = python_command + ' ' + script_name + ' ' + config_filename + ' ' + nxs_filename + ' ' + output_dir + ' ' + str(run_number)
                command_mtz = python_command + ' ' + mtz_script + ' ' + config_filename + ' ' + output_dir + ' ' + str(run_number)
                with open(STATUS_FILE, 'w') as f:
                    f.write('Processing run {0}\n{1}\n'.format(NEXT_RUN_TO_PROCESS, ipts_number))
                subprocess.call(command_pf.split(' ')) #Does the profile fitting
                subprocess.call(command_mtz.split(' '))
                with open(STATUS_FILE, 'w') as f:
                    f.write('Finished processing run {0}. Waiting...\n{1}\n'.format(NEXT_RUN_TO_PROCESS, ipts_number))
                print(command_pf)
                print(command_mtz)
            write_last_processed_run(NEXT_RUN_TO_PROCESS)
            LAST_PROCESSED_RUN += 1
            NEXT_RUN_TO_PROCESS += 1
    except KeyboardInterrupt:
        sys.exit(1)

