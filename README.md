# MaNDi Autoreduction Scripts

## Autoreduction Instructions
### General
This repository contains a series of scripts that collectively run the autoreduction for the MaNDi beamline at the SNS. The setup is relatively hacky and users should take time to verify that the output makes sense.  It is designed to run on a single machine (e.g. `mandi1.sns.gov`) and checks the specified IPTS folder for a new nexus file every minute.  When a new file is found, it will check if a config file (must be called `/SNS/MANDI/IPTS-YYYY/shared/autoreduce/mandi_autoreduce.config` where `YYYY` is the IPTS number.)  In addition, the process must be started by a user who has write access to the IPTS experiment and /SNS/MANDI/shared/ (typically beamline staff).

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
(This feature must be added still).

### Output from Autoreduction
Autoreduction will output a few files per run.  All of these files are saved to `/SNS/MANDI/IPTS-YYYY/shared/autoreduce/`.  A description is given below (for run number XXXX):

1) `peaks_profileFitted_ws_XXXX_mandi_autoreduced.integrate`: A peaks workspace containing the intensities and sigmas from profile fitting.

2) `peaks_ws_XXXX_mandi_autoreduced.integrate`: A peaks workspace containing the spherically integrated intensities and sigmas.

3) `params_ws_XXXX_mandi_autoreduced.nxs`: A workspace containing the summary of fit parameters for each peak.  No entry is present for peaks which fail to fit.

4) `UB_ws_XXXX_mandi_autoreduced`: The UB Matrix from each run.

5) `XXXX_fig[1-5].png`: Figures summarizing the fits.  These images can be useful for quick diagnostics of fit quality.

6) `laue/`: This directory contains the files used to run `lauenorm`.  It is probably a good idea to inspect the logs (`lnorms70aMaNDi.log` and `lnorms70aMaNDi_merged.log`) to make sure `lauenorm` is behaving as expected.  Intensites are written to `laueNorm*` files.  The resulting mtz files contain merged and unmerged intensities which can be directly used in phenix.
