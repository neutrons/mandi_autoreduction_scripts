# MaNDi Autoreduction Scripts

## Autoreduction Instructions
### General
This repository contains a series of scripts that collectively run the autoreduction for the MaNDi beamline at the SNS. The setup is relatively hacky and users should take time to verify that the output makes sense.  It is designed to run on a single machine (e.g. the MaNDi analysis machines) and checks the specified IPTS folder for a new nexus file every minute.  When a new file is found, it will check if a config file (must be called `/SNS/MANDI/IPTS-YYYY/shared/autoreduce/mandi_autoreduce.config` where `YYYY` is the IPTS number.)  In addition, the process must be started by a user who has write access to the IPTS experiment and /SNS/MANDI/shared/ (typically beamline staff).

### Setting up a New Experiment
1) If autoreduction is currently running, kill it.  This can be done by killing any terminal that is running `start_run_checker.py`

2) Copy an old `.config` file to the current IPTS autoreduce directory.  A template is stored at `/SNS/MANDI/shared/autoreduce/mandi_autoreduce.config`.  The file *must* be named `mandi_autoreduce.config`.  So, for `IPTS-YYYY` run the following command in a terminal:

```bash
$ cp /SNS/MANDI/shared/autoreduce/mandi_autoreduce.config /SNS/MANDI/IPTS−YYYY/shared/autoreduce/
```

3) Modify `/SNS/MANDI/IPTS−YYYY/shared/autoreduce/mandi_autoreduce.config` to match your experiment.  The file is commented to explain each parameter.

4) Update the file `/SNS/MANDI/shared/autoreduce/last_run_processed.dat`.  This file contains the run number of the last analyzed run.  Autoreduction will keep looking for the output of run number 1+this number.  (Note that this number increments after each run.  So if you are processing sequential runs you need only do this once per experiment.)

5) Update the file `/SNS/MANDI/shared/autoreduce/autoreduce_status.txt`.  The first line is the status - only used for monitoring what's going on.  The second line is the IPTS number being monitored.  This number must be updated for each experiment.

6) Start the autoreduction software.  The `start_run_checker.py` will monitor for new runs and set them processing.  To run this:

```bash
$ /SNS/MANDI/shared/autoreduce/start_run_checker.py
```

7) Optionally, start the monitor.  This is just a small GUI that shows what autoreduction is doing.  This can be run from any machine that can read `/SNS/MANDI/shared/autoreduce` by running

```bash
$ /SNS/MANDI/shared/autoreduce/autoreduce_monitor.py
```

### Re-analyzing Runs
Currently, autoreduction only looks for the event (`.nxs`) file from the run after the run number in `/SNS/MANDI/shared/autoreduce/last_run_processed.dat`.  So to re-analyze a single run, just follows the steps to start autoreduction setting the appropriate run number.

### Analyzing series of runs by hand
To analyze a set of runs by hand (i.e. when revisiting old data), use the scripts in the `manual_reduction` folder.  This folder has three files.  You should copy these three folders to a working directory.  The files are:

1) `mandi_parallel.py`: This script should be all you need to change for most experiments.  It defines the parameters for integration (which runs to integrate, sample dimensions, etc.) and creates a dictionary of parameters for each run to be analyzed.  Each parameter is commented with an explanation.  After saving the dictionary files, this program will run the integration for each run in `run_nums`.  It will run upto `max_processes` at a time, starting the next run as runs finish.  To run this:

```bash
$ python mandi_parallel.py
```

2) `mandi_singleRun.py`: This script does the "heavy lifting" for integration.  It loads the run (and DetCal file optionally), converts to reciprocal space, finds peaks, indexes peaks, integrates them using spherical integration, and then integrates using profile fitting.  The script will write four files at the end of each run: two `.integrate` files for spherical and profile fitted intensities, a `.mat` file containing the UB matrix used for indexing, and a `.nxs` file containing the parameters of each peak's fit.  This script should not need to be run manually.

3) `mandi_createmtz.py`: This script loads the runs, filters peaks, outputs the input files for laueNorm, and runs laueNorm resulting in merged and unmerged mtz files.  It should be run after mandi_parallel.py has finished.  To run this script:

```bash
$ mantidpython mandi_createmtz.py fitParams_XXXX.pkl path/to/working/directory XXXX
```
where XXXX is the highest run number being analyzed.  (Choosing the highest isn't that important, but for easy use with autoreduction it needs a run number - in reality it will base everything on the list of run numbers in the pkl file.)  Note that the config file **must end with .pkl** for manual reduction.  Output files, including the mtz files and logs will be stored in `path/to/working/directory/laue/`.

### Output from Autoreduction
Autoreduction will output a few files per run.  All of these files are saved to `/SNS/MANDI/IPTS-YYYY/shared/autoreduce/`.  A description is given below (for run number XXXX):

1) `peaks_profileFitted_ws_XXXX_mandi_autoreduced.integrate`: A peaks workspace containing the intensities and sigmas from profile fitting.

2) `peaks_ws_XXXX_mandi_autoreduced.integrate`: A peaks workspace containing the spherically integrated intensities and sigmas.

3) `params_ws_XXXX_mandi_autoreduced.nxs`: A workspace containing the summary of fit parameters for each peak.  No entry is present for peaks which fail to fit.

4) `UB_ws_XXXX_mandi_autoreduced`: The UB Matrix from each run.

5) `XXXX_fig[1-5].png`: Figures summarizing the fits.  These images can be useful for quick diagnostics of fit quality.

6) `laue/`: This directory contains the files used to run `lauenorm`.  It is probably a good idea to inspect the logs (`lnorms70aMaNDi.log` and `lnorms70aMaNDi_merged.log`) to make sure `lauenorm` is behaving as expected.  Intensites are written to `laueNorm*` files.  The resulting mtz files contain merged and unmerged intensities which can be directly used in phenix.
