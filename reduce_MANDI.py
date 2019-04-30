import sys
import os
import numpy as np
import matplotlib
matplotlib.use("agg")

if __name__ == "__main__":
    np.seterr("ignore")  # ignore division by 0 warning in plots
    # check number of arguments
    if (len(sys.argv) != 3):
        logger.error("autoreduction code requires a filename and an output directory")
        sys.exit()
    if not(os.path.isfile(sys.argv[1])):
        logger.error("data file "+sys.argv[1]+ " not found")
        sys.exit()    
    else:
        filename = sys.argv[1]
        outdir = sys.argv[2]
        load_venv_command = 'source /SNS/snfs1/instruments/MANDI/shared/autoreduce/venv/bin/activate; '
        python_command = '/SNS/users/ntv/workspace/mantid/release/bin/mantidpython'
        script_to_call = '/SNS/snfs1/instruments/MANDI/shared/autoreduce/reduce_MANDI_doWork.py'
        total_command = load_venv_command + python_command + ' ' + script_to_call + ' ' + filename + ' '+ outdir
        #os.system(total_command)
