from __future__ import (absolute_import, division, print_function)
import os
import threading
import time
import pickle


#-------------------------------------------------------------------parameters in this block need to be set.
# this should point to the version of python you'd like to run.  Until the algorith is more stable, this will
# probably be the nightly or a dev build (on analysis, you probably want mantidpythonnightly)
python_command = 'mantidpythonnightly'
peaksFile = None
UBFile = None
# parameters for parallel runs
outDir = '/SNS/users/USR/Desktop/beta_lac_july2018_secondtxtal/' #Where we save results - should end with /
max_processes = 3 #Largest number of runs to reduce at a time - typically limited by RAM.
reduce_one_run_script = 'mandi_singleRun.py'
eventFilesFormat = '/SNS/MANDI/IPTS-8776/nexus/MANDI_%i.nxs.h5' # %i will be replaced by the run number
run_nums = range(9113,9117+1) #Which run numbers to analyze - must be python iterable (e.g. list, so you can use range)
                              #If using range(), remember you add 1 to the final number you want processed.

# Spherical integration parameters - should be close to reasonable if you want
# a comparison with profile fitting
rA = 0.020 #Inner radius for spherical integration
rB = 0.021 #Inner radius for background shell for spherical integration
rC = 0.023 #Outer radius for background shell for spherical integration

# For peak finding, predicting, and indexing
minD = 60 #Estimate of smallest unit cell side length
maxD = 110 #Estimate of largest unit cell side length
tol = 0.15 #Tolerance for indexing
a, b, c = 73.3, 73.3, 99.0 #Unit cell lenghts in angstrom
alpha, beta, gamma = 90, 90, 120 #Angles for unit cell in degrees
predictPeaks = True #Boolean to predict peaks or not
min_pred_dspacing = 1.8 #Lowest d to predict peaks to
max_pred_dspacing = 15.0 #Highest d to predict peak to
min_pred_wl = 2.0 #Low wavelength to predict peaks
max_pred_wl = 4.0 #High wavelength to predict peaks

# Parameters for lauenorm
lauenormSetupScript = '/SNS/snfs1/instruments/MANDI/shared/laue3/laue/laue.setup-sh' #location of the laue.setup-sh for lauegen
spacegroup_number = 154 #Space group number for laueNorm.
mtz_name = 'betalac' #name for mtz files resulting from laueNorm 
lauenorm_edge_pixels = 1 #number of pixels from the edge to exclude
lauenorm_scale_peaks = 3. #Value to scale peak intensitys and sigmas by
lauenorm_min_d = 1.2 #Minimum d value to output for laueNorm
lauenorm_min_wl = 2.1 #Min wavelength to output for laueNorm
lauenorm_max_wl = 3.9 #Max wavelength to output for laueNorm
lauenorm_min_isi = 1.0 #Min I/sigma to output for laueNorm
lauenorm_mini = 0.0 #Min I to output for laueNorm
lauenorm_applysinsq = True #If true, will multiply intensities and sigmas by sin(theta)**2
                           #Allowing lauenorm to apply the wavelength component of the Lorentz
                           #correction.
force_lattice_parameters = False #When indexing all runs together, should we run FindUBUsingLatticeParameters.
                                 #If this is false, you will be asked if you want to after an optimal UB is found
                                 #in mandi_createmtz.py, since sometimes this still finds the Niggli cell.

# Directories for lauenorm
pbpDir = '/SNS/MANDI/shared/laue3/' #pbp directory for laueNorm.  Must contain multidiags.out (from laugen). Should end with /
laueLibDir = '/SNS/MANDI/shared/laue3/laue/laue_install/lib/' #library file for lauegen.  Should end with /
lauenormBin = '/SNS/MANDI/shared/laue3/laue/laue_install/bin/lauenorm' #Binary file for lauenorm.


# IntegratePeaksProfileFitting parameters
ModeratorFile = '/SNS/users/USR/integrate/bl11_moderatorCoefficients_2018.dat' #file with moderator coefficients
StrongPeaksParamsFile = None #string to pkl file containing strong peaks parameters
DetCalFile = '/SNS/MANDI/shared/ProfileFitting/MANDI_June2018.DetCal'
IntensityCutoff = 200 #Peaks with spherical intensity below IntensityCutoff are considered weak peaks
EdgeCutoff = 3 #Peaks within EdgeCutoff of the edge will have their profiles taken from a strong peak

# --(the parameters here probably don't need changing)--
FracStop = 0.15 #Fraction of the peak maximum we use to define the peak volume
MinpplFrac = 0.9 #Lower fraction of background estimate to check against expected TOF profile
MaxpplFrac = 1.1 #High  fraction of background estimate to check against expected TOF profile
DQMax = 0.15 #Maximum box side length to consider for fitting.
#-------------------------------------------------------------------end of parameters block

# Create a dictionary for parameters
d = {}
d['outDir'] = outDir
d['max_processes'] = max_processes
d['reduce_one_run_script'] = reduce_one_run_script
d['eventFilesFormat'] = eventFilesFormat
d['run_nums'] = run_nums
d['rA'] = rA
d['rB'] = rB
d['rC'] = rC
d['minD'] = minD
d['maxD'] = maxD
d['tol'] = tol
d['ModeratorFile'] = ModeratorFile
d['StrongPeaksParamsFile'] = StrongPeaksParamsFile
d['IntensityCutoff'] = IntensityCutoff
d['EdgeCutoff'] = EdgeCutoff
d['FracStop'] = FracStop
d['MinpplFrac'] = MinpplFrac
d['MaxpplFrac'] = MaxpplFrac
d['DQMax'] = DQMax
d['predictPeaks'] = predictPeaks
d['min_pred_dspacing'] = min_pred_dspacing
d['max_pred_dspacing'] = max_pred_dspacing
d['min_pred_wl'] = min_pred_wl
d['max_pred_wl'] = max_pred_wl
d['a'] = a
d['b'] = b
d['c'] = c
d['alpha'] = alpha
d['beta'] = beta
d['gamma'] = gamma
d['peaksFile'] = peaksFile
d['UBFile'] = UBFile
d['DetCalFile'] = DetCalFile
d['spacegroup_number'] = spacegroup_number
d['mtz_name'] = mtz_name
d['lauenorm_edge_pixels'] = lauenorm_edge_pixels
d['lauenorm_scale_peaks'] = lauenorm_scale_peaks
d['lauenorm_min_d'] = lauenorm_min_d
d['lauenorm_min_wl'] = lauenorm_min_wl
d['lauenorm_max_wl'] = lauenorm_max_wl
d['lauenorm_min_isi'] = lauenorm_min_isi
d['lauenorm_mini'] = lauenorm_mini
d['lauenorm_applysinsq'] = lauenorm_applysinsq
d['pbpDir'] = pbpDir
d['laueLibDir'] = laueLibDir
d['lauenormBin'] = lauenormBin
d['force_lattice_parameters'] = force_lattice_parameters
d['lauenormSetupScript'] = lauenormSetupScript

#Define a class for threading (taken from ReduceSCD_Parallel.py) and set up parallel runs
class ProcessThread ( threading.Thread ):
    command = ""

    def setCommand( self, command="" ):
        self.command = command

    def run ( self ):
        print('STARTING PROCESS: ' + self.command)
        os.system( self.command )


python = python_command
procList=[]
index = 0
for r_num in run_nums:
    dictPath = outDir + 'fitParams_%i.pkl'%r_num
    d['runNumber'] = r_num
    pickle.dump(d, open(dictPath, 'wb'))
    procList.append( ProcessThread() )
    cmd = '%s %s %s' % (python, reduce_one_run_script, dictPath)
    procList[index].setCommand( cmd )
    index = index + 1

#
# Now create and start a thread for each command to run the commands in parallel,
# starting up to max_processes simultaneously.
#
all_done = False
active_list=[]
while not all_done:
    if  len(procList) > 0 and len(active_list) < max_processes :
        thread = procList[0]
        procList.remove(thread)
        active_list.append( thread )
        thread.start()
    time.sleep(2)
    for thread in active_list:
        if not thread.isAlive():
            active_list.remove( thread )
    if len(procList) == 0 and len(active_list) == 0 :
        all_done = True
