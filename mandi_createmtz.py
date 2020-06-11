from __future__ import (absolute_import, division, print_function)
import os
import sys
import subprocess
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
from mantid.simpleapi import *
from mantid.api import *
import ReduceDictionary
import pandas as pd
import numpy as np
import glob
import h5py
import pickle

def readParamsNexusFile(paramFileName):
    """
    This function reads the parameters output from IntegratePeaksProfileFitting.
    Loading them through mantid takes about 20 times longer than using h5py directly,
    so we do it this way.
    """
    d = {}
    with h5py.File(paramFileName) as f:
        colsToRead = f['mantid_workspace_1/table_workspace/'].keys()
        for col in colsToRead:
            key = 'mantid_workspace_1/table_workspace/{}'.format(col)
            colName = str(f[key].attrs['name'].decode('utf8'))
            colVal = f[key].value
            if colName != 'newQ':
                d[colName] = colVal
            else:  # This skips newQ because creating a dataframe
                newQList = []
                for v in colVal:
                    newQList.append(v)
                d[colName] = newQList
    df = pd.DataFrame(d)
    return df

def parsePickledParameters(paramFileName):
    """
    This function is a converter from the manual parameters dictionary ("pkl")
    to the parameters dictionary defined by the *config file used for autoreduction.
    """
    dIn = pickle.load(open(paramFileName, 'rb'))
    d = {} #Output dictionary, in the format of a .config file
    d['unitcell_a'] = dIn['a']
    d['unitcell_b'] = dIn['b']
    d['unitcell_c'] = dIn['c']
    d['unitcell_alpha'] = dIn['alpha']
    d['unitcell_beta'] = dIn['beta']
    d['unitcell_gamma'] = dIn['gamma']
    d['first_run_number'] = np.min(dIn['run_nums'])
    d['spacegroup_number'] = dIn['spacegroup_number']
    d['mtz_name'] = dIn['mtz_name']
    d['lauenorm_edge_pixels'] = dIn['lauenorm_edge_pixels']
    d['lauenorm_scale_peaks'] = dIn['lauenorm_scale_peaks']
    d['lauenorm_min_d'] = dIn['lauenorm_min_d']
    d['lauenorm_min_wl'] = dIn['lauenorm_min_wl']
    d['lauenorm_max_wl'] = dIn['lauenorm_max_wl']
    d['lauenorm_min_isi'] = dIn['lauenorm_min_isi']
    d['lauenorm_mini'] = dIn['lauenorm_mini']
    d['lauenorm_applysinsq'] = dIn['lauenorm_applysinsq']
    d['pbpDir'] = dIn['pbpDir']
    d['laueLibDir'] = dIn['laueLibDir']
    d['lauenormBin'] = dIn['lauenormBin']
    d['tolerance'] = dIn['tol']
    d['force_lattice_parameters'] = dIn['force_lattice_parameters']
    d['run_nums'] = dIn['run_nums']
    d['lauenormSetupScript'] = dIn['lauenormSetupScript']
    return d

def createMTZFile(d, out_dir, run_number, run_numbers_to_process=None):
    """
    Puts all runs between 
    """
    a = float(d['unitcell_a'])
    b = float(d['unitcell_b'])
    c = float(d['unitcell_c'])
    alpha = float(d['unitcell_alpha'])
    beta = float(d['unitcell_beta'])
    gamma = float(d['unitcell_gamma'])
    first_run_number = int(d['first_run_number'])
    spacegroup_number = int(d['spacegroup_number'])
    mtz_name = d['mtz_name']
    lauenorm_edge_pixels = int(d['lauenorm_edge_pixels'])
    lauenorm_scale_peaks = float(d['lauenorm_scale_peaks'])
    lauenorm_min_d = float(d['lauenorm_min_d'])
    lauenorm_min_wl = float(d['lauenorm_min_wl'])
    lauenorm_max_wl = float(d['lauenorm_max_wl'])
    lauenorm_min_isi = float(d['lauenorm_min_isi'])
    lauenorm_mini = float(d['lauenorm_mini'])
    lauenorm_applysinsq = bool(d['lauenorm_applysinsq'])
    pbpDir = d['pbpDir']
    laueLibDir = d['laueLibDir']
    lauenormBin = d['lauenormBin']
    lauenormSetupScript = d['lauenormSetupScript']
    tolerance = float(d['tolerance'])
    force_lattice_parameters = bool(d['force_lattice_parameters'])
    laue_directory = out_dir + 'laue/'
    laue_directory = os.path.abspath(laue_directory) + '/'

    # Create the combined workspaces and a pandas dataframe that
    # we can use to filter bad fits.
    outputFilenameTemplate = out_dir + '%s_ws_%i_mandi_autoreduced.%s'
    runNumbersProcessed = []
    dfList = []

    # If we are not given an explicit list of run numbers, we assume
    # that we should go from first_run_number to run_number.
    if run_numbers_to_process is None:
        run_numbers_to_process = range(first_run_number, run_number + 1)
    for rn in run_numbers_to_process:
        print('createMTZ - starting run %i' % rn)
        paramsFileName = outputFilenameTemplate % ('params', rn, 'nxs')
        peaksFileName = outputFilenameTemplate % ('peaks', rn, 'integrate')
        peaksPFFileName = outputFilenameTemplate % ('peaks_profileFitted',
                                                    rn, 'integrate')
        matFileName = outputFilenameTemplate % ('UB', rn, 'mat')
        paramsFileExists = os.path.isfile(paramsFileName)
        peaksExists = os.path.isfile(peaksFileName)
        peaksPFExists = os.path.isfile(peaksPFFileName)
        matExists = os.path.isfile(matFileName)
        if(paramsFileExists and peaksExists and peaksPFExists and matExists):
            logger.information('Including run number {0:d}'.format(rn))
            runNumbersProcessed.append(rn)
            peaks_ws = LoadIsawPeaks(Filename=peaksFileName)
            peaks_ws_profile = LoadIsawPeaks(Filename=peaksPFFileName)

            dfTWS = pd.DataFrame(peaks_ws.toDict())
            dfTParams = readParamsNexusFile(paramsFileName)
            dfT = pd.merge(dfTWS, dfTParams, left_on='PeakNumber',
                           right_on='peakNumber', how='outer')
            dfT = dfT[~pd.isnull(dfT['QLab'])]
            dfT['theta'] = dfT['QLab'].apply(lambda x: np.arctan2(
                                             x[2], np.hypot(x[0], x[1])))
            dfT['phi'] = dfT['QLab'].apply(lambda x: np.arctan2(x[1], x[0]))
            dfList.append(dfT)

            if len(runNumbersProcessed) == 1:  # First peak we've added
                pwsSPH = CloneWorkspace(InputWorkspace=peaks_ws,
                                        OutputWorkspace='pwsSPH')
                pwsPF = CloneWorkspace(InputWorkspace=peaks_ws_profile,
                                       OutputWorkspace='pwsPF')
            else:  # Append the current workspaces
                pwsSPH = CombinePeaksWorkspaces(LHSWorkspace=pwsSPH,
                                                RHSWorkspace=peaks_ws,
                                                OutputWorkspace=pwsSPH)
                pwsPF = CombinePeaksWorkspaces(LHSWorkspace=pwsPF,
                                               RHSWorkspace=peaks_ws_profile,
                                               OutputWorkspace=pwsPF)
        else:
            print('Cannot find one of the following files.'
                  '  Will continue without this run:')
            print(paramsFileName, peaksFileName, peaksPFFileName) 
        print('createMTZ - finished run %i' % rn)
    if (len(dfList) > 0):
        df = pd.concat(dfList)
        df = df.reset_index()
    else:
        logger.error('No runs to be added to create the mtz file!  Exiting!')
        sys.exit()

    # Create graphs which can be displayed on monitor
    gIDX = (df['chiSq'] < 50) & (df['chiSq3d'] < 10)

    plt.figure(1, figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df[gIDX]['Intens'], df[gIDX]['Intens3d'], '.', ms=2)
    plt.plot([1, df[gIDX]['Intens'].max()], [1, df[gIDX]['Intens'].max()],
             alpha=0.8)
    plt.xlabel('Spherical Integration Intensity')
    plt.ylabel('Profile Fitted Intensity')
    plt.title('Intensities')
    plt.subplot(1, 2, 2)
    plt.plot(df['Energy'], df['T0'], '.', ms=1.5, label='T0')
    plt.legend(loc='best')
    plt.xlabel('Energy (meV)')
    plt.ylabel('T0 (us)')
    plt.title('T0 vs Energy')
    plt.savefig(out_dir + '{0:d}_fig1.png'.format(run_number))

    # Now let's check out the I-C parameters
    # We expect energy dependence but no angular dependence
    gIDX = (df['chiSq'] < 50) & (df['chiSq3d'] < 10)
    strongIDX = gIDX & (df['Intens'] > 200) & (df['Intens3d'] > 200)

    plt.figure(2, figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df[gIDX]['Energy'], df[gIDX]['Alpha'], '.',
             ms=1.5, label='Alpha', alpha=0.2)
    plt.plot(df[gIDX]['Energy'], df[gIDX]['Beta'], '.',
             ms=1.5, label='Beta', alpha=0.2)
    plt.plot(df[gIDX]['Energy'], df[gIDX]['R'], '.',
             ms=1.5, label='R', alpha=0.2)
    plt.legend(loc='best')
    plt.xlabel('Energy (meV)')
    plt.ylabel('Value')
    plt.title('All Peaks')
    plt.subplot(1, 2, 2)
    plt.plot(df[strongIDX]['Energy'], df[strongIDX]['Alpha'], '.',
             ms=1.5, label='Alpha', alpha=0.2)
    plt.plot(df[strongIDX]['Energy'], df[strongIDX]['Beta'], '.',
             ms=1.5, label='Beta', alpha=0.2)
    plt.plot(df[strongIDX]['Energy'], df[strongIDX]['R'], '.',
             ms=1.5, label='R', alpha=0.2)
    plt.legend(loc='best')
    plt.xlabel('Energy (meV)')
    plt.ylabel('Value')
    plt.title('Strong Peaks')
    plt.savefig(out_dir + '{0:d}_fig2.png'.format(run_number))

    plt.figure(3, figsize=(12, 4))
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(df[gIDX]['theta'], df[gIDX]['SigX'], '.',
             ms=1, alpha=0.3)
    plt.xlabel('Theta (along scattering direction) (rad)')
    plt.ylabel('Sigma Scattering (rad)')
    plt.title('All Peaks')
    plt.subplot(1, 2, 2)
    plt.plot(df[strongIDX]['theta'], df[strongIDX]['SigX'], '.',
             ms=1, alpha=0.3)
    plt.xlabel('Theta (along scattering direction) (rad)')
    plt.ylabel('Sigma Scattering (rad)')
    plt.title('Strong Peaks')
    plt.savefig(out_dir + '{0:d}_fig3.png'.format(run_number))

    plt.figure(4, figsize=(12, 4))
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(df[gIDX]['phi'], df[gIDX]['SigY'], '.',
             ms=1, alpha=0.2)
    plt.xlabel('Phi_azimuthal (rad)')
    plt.ylabel('Sigma azimuthal (rad)')
    plt.title('All Peaks')
    plt.subplot(1, 2, 2)
    plt.plot(df[strongIDX]['phi'], df[strongIDX]['SigY'], '.',
             ms=1, alpha=0.2)
    plt.xlabel('Phi_azimuthal (rad)')
    plt.ylabel('Sigma azimuthal (rad)')
    plt.title('Strong Peaks')
    plt.savefig(out_dir + '{0:d}_fig4.png'.format(run_number))

    # Reindex to make sure everything is in the same coordinate system
    numPeaksIndexed = np.zeros_like(runNumbersProcessed)
    for i, runNumber in enumerate(runNumbersProcessed):
        UBFileName = outputFilenameTemplate % ('UB', runNumber, 'mat')
        LoadIsawUB(InputWorkspace=pwsPF, Filename=UBFileName)
        numIndexed = IndexPeaks(pwsPF, tolerance=tolerance)[0]
        numPeaksIndexed[i] = numIndexed
    gIDX = np.argmax(numPeaksIndexed)
    LoadIsawUB(InputWorkspace=pwsPF, Filename=outputFilenameTemplate % (
               'UB', runNumbersProcessed[gIDX], 'mat'))
    LoadIsawUB(InputWorkspace=pwsSPH, Filename=outputFilenameTemplate % (
               'UB', runNumbersProcessed[gIDX], 'mat'))
    numIndexed = IndexPeaks(pwsPF, tolerance=tolerance)[0]
    numIndexed = IndexPeaks(pwsSPH, tolerance=tolerance)[0]
    percentIndexed = 100. * numPeaksIndexed[gIDX] / pwsPF.getNumberPeaks()
    print('There are {0:d} peaks total.  The UB matrix from run {1:d}'
          '  will index {2:d} of them ({3:4.2f} percent).'
          '  Using this file.'.format(pwsPF.getNumberPeaks(),
                                      runNumbersProcessed[gIDX],
                                      numPeaksIndexed[gIDX],
                                      percentIndexed))
    lattice = pwsPF.sample().getOrientedLattice()

    if not force_lattice_parameters:
        print('The optimized lattice is', lattice)
        print('The input lattice was ', a,b,c, alpha, beta, gamma)
        strRedoIndex = raw_input('Would you like to try reindexing using FindUBUsingLatticeParameters? (Y/[n])  ')
        if 'Y' in strRedoIndex.upper():
            force_lattice_parameters = True
    if force_lattice_parameters:
        print('Reindexing in new coordinate system.')
        print('This may take serveral minutes.')
        FindUBUsingLatticeParameters(PeaksWorkspace=pwsPF, a=a, b=b, c=c,
                                     alpha=alpha, beta=beta, gamma=gamma,
                                     NumInitial=50, Tolerance=tolerance,
                                     Iterations=1000)
        FindUBUsingLatticeParameters(PeaksWorkspace=pwsSPH, a=a, b=b, c=c,
                                     alpha=alpha, beta=beta, gamma=gamma,
                                     NumInitial=50, Tolerance=tolerance,
                                     Iterations=1000)
        numIndexed = IndexPeaks(PeaksWorkspace=pwsPF)[0]
        numIndexed = IndexPeaks(PeaksWorkspace=pwsSPH)[0]
        lattice = pwsPF.sample().getOrientedLattice()
        print('New lattice:')
        print(lattice)
        print('Indexes {0:d} of {1:d} peaks'.format(numIndexed,
                                                    pwsPF.getNumberPeaks()))

    df['h_reindexed'] = pwsPF.column('h')
    df['k_reindexed'] = pwsPF.column('k')
    df['l_reindexed'] = pwsPF.column('l')

    # Write our mtz files
    goodIDX = (df['chiSq'] < 50.0) & (df['chiSq3d'] < 10)
    edgeIDX = ((df['Row'] <= lauenorm_edge_pixels) |
               (df['Row'] >= 255 - lauenorm_edge_pixels) |
               (df['Col'] <= lauenorm_edge_pixels) |
               (df['Col'] >= 255 - lauenorm_edge_pixels))
    print('Rejecting {0} peaks for bad fits and {1} peaks '
          'for being on the edge'.format(np.sum(~goodIDX), np.sum(edgeIDX)))
    goodIDX = goodIDX & ~edgeIDX

    # Apply sin(theta)**2, lauenorm does wavelength part of Lorentz correction
    if lauenorm_applysinsq:
        df['lorentzFactor'] = df['theta'].apply(lambda x: 1000 * np.sin(x)**2)
        df['Intens3d_normalized'] = df['Intens3d'] * df['lorentzFactor']
        df['SigInt3d_normalized'] = df['SigInt3d'] * df['lorentzFactor']
    else:
        df['Intens3d_normalized'] = df['Intens3d']
        df['SigInt3d_normalized'] = df['SigInt3d']

    ws = CloneWorkspace(InputWorkspace=pwsPF, OutputWorkspace='ws')
    ws2 = CloneWorkspace(InputWorkspace=pwsSPH, OutputWorkspace='ws2')
    for i in range(len(df)):
        if goodIDX[i]:
            newI = float(df.iloc[i]['Intens3d_normalized'])
            newSig = float(df.iloc[i]['SigInt3d_normalized'])
            ws.getPeak(i).setIntensity(newI)
            ws.getPeak(i).setSigmaIntensity(newSig)
        else:
            ws.getPeak(i).setIntensity(lauenorm_mini - 1.)
            ws.getPeak(i).setSigmaIntensity(1.0)
            ws2.getPeak(i).setIntensity(lauenorm_mini - 1.)
            ws2.getPeak(i).setSigmaIntensity(1.0)

    plt.figure()
    plt.clf()
    plt.plot(ws2.column('Intens'), ws.column('Intens'), '.', ms=1)
    plt.xlabel('Spherical Intensity')
    plt.ylabel('Profile Fitted Intensity')
    plt.title('Intensities to be output for lauenorm')
    plt.savefig(out_dir + '{}_fig5.png'.format(run_number))
    oldLaueNormFiles = glob.glob(laue_directory + 'laueNorm*')
    for fileName in oldLaueNormFiles:
        os.remove(fileName)
    print('********************')
    print(laue_directory)
    print(lauenorm_scale_peaks, lauenorm_min_d,lauenorm_min_wl,lauenorm_max_wl,lauenorm_min_isi,lauenorm_mini)
    SaveLauenorm(InputWorkspace=ws, Filename=laue_directory + 'laueNorm',
                 ScalePeaks=lauenorm_scale_peaks, MinDSpacing=lauenorm_min_d,
                 MinWavelength=lauenorm_min_wl, MaxWavelength=lauenorm_max_wl,
                 SortFilesBy='RunNumber', MinIsigI=lauenorm_min_isi,
                 MinIntensity=lauenorm_mini)
    print('Wrote laueNorm input files to %s' % (laue_directory))

    comFilename = laue_directory + 'lnorm.com'
    datFilename = laue_directory + 'lnorm.dat'
    datFilenameMerged = laue_directory + 'lnorm_merged.dat'
    numRuns = len(np.unique(ws.column('RunNumber')))
    lattice = pwsPF.sample().getOrientedLattice()

    # unmerged .dat file
    with open(datFilename, 'w') as f:
        f.write('5s70aMaNDi3\n')
        f.write('%2.2f %2.2f %2.2f %i %i %i\n' % (
                lattice.a(), lattice.b(), lattice.c(),
                np.round(lattice.alpha()),
                np.round(lattice.beta()),
                np.round(lattice.gamma())))
        f.write('NORMALISE %i\n' % numRuns)
        f.write('UNITY\n')
        f.write('SYMM 0.1\n')
        f.write('%i 1 8 8 1 4 1\n' % spacegroup_number)
        f.write('1 1 1 %4.1f 0 0 0 2\n' % lauenorm_min_isi)
        f.write('%1.1f %1.1f 10 6 3\n' % (lauenorm_min_wl, lauenorm_max_wl))
        f.write('3\n')
        f.write('0 25.0 0 0 0')
    print('Wrote unmerged lauenorm configuration to %s' % datFilename)
    # merged .dat file
    with open(datFilenameMerged, 'w') as f:
        f.write('5s70aMaNDi3\n')
        f.write('%2.2f %2.2f %2.2f %i %i %i\n' % (
                lattice.a(), lattice.b(), lattice.c(),
                np.round(lattice.alpha()),
                np.round(lattice.beta()),
                np.round(lattice.gamma())))
        f.write('NORMALISE %i\n' % numRuns)
        f.write('UNITY\n')
        f.write('SYMM 0.1\n')
        f.write('%i 1 8 8 1 4 1\n' % spacegroup_number)
        f.write('1 1 1 %4.1f 0 0 0 2\n' % lauenorm_min_isi)
        f.write('%1.1f %1.1f 10 6 3\n' % (lauenorm_min_wl, lauenorm_max_wl))
        f.write('1\n')
        f.write('0 25.0 0 0 0')
    print('Wrote merged lauenorm configuration to %s' % datFilenameMerged)
    # executable
    with open(comFilename, 'w') as f:
        f.write('#!/bin/sh\n')
        f.write('source %s\n' % lauenormSetupScript)  # noqa: E501
        f.write('cwd=$(pwd)\n')
        for runNum in range(numRuns):
            f.write('LAUE%03i=$cwd/laueNorm%03i\n' % (runNum + 1, runNum + 1))
        f.write('\n\n')
        f.write('HKLOUT=$cwd/%s_unmerged.mtz\n' % mtz_name)
        f.write('HKLMULT=%shklmult_image.out\n' % pbpDir)
        f.write('MULTDIAG=%smultidiags.out\n' % pbpDir)
        f.write('PGDATA=%spglib.dat\n' % laueLibDir)
        f.write('SYMOP=%ssymop.lib\n' % laueLibDir)
        f.write('SYMINFO=%ssyminfo.lib\n' % laueLibDir)
        for runNum in range(numRuns):
            f.write('export LAUE%03i\n' % (runNum + 1))
        f.write('\n')
        f.write('export HKLOUT\n')
        f.write('export HKLMULT\n')
        f.write('export MULTDIAG\n')
        f.write('export PGDATA\n')
        f.write('export SYMOP\n')
        f.write('export SYMINFO\n')
        f.write('time %s < %slnorm.dat > %slnorms70aMaNDi.log\n' % (
                lauenormBin, laue_directory, laue_directory))
        f.write('HKLOUT=$cwd/%s_merged.mtz\n' % mtz_name)
        f.write('export HKLOUT\n')
        f.write('time %s < %slnorm_merged.dat > %slnorms70aMaNDi_merged.log\n' % (  # noqa: E501
                lauenormBin, laue_directory, laue_directory))
    os.chmod(comFilename, 0o775)
    print('Wrote lauenorm executable to %s' % comFilename)
    print('Running laueNorm...')
    mtd.clear()
    subprocess.Popen(comFilename,
                     cwd=os.path.dirname(os.path.realpath(comFilename)))


if __name__ == '__main__':
    if (len(sys.argv) != 4):
        logger.error('{0} must take 3 arguments!'.format(sys.argv[0]))
        sys.exit()
    if not(os.path.isfile(sys.argv[1])):
        logger.error("config file " + sys.argv[1] + " not found")
        sys.exit()
    else:
        config_file_name = str(sys.argv[1])
        out_dir = os.path.abspath(str(sys.argv[2])) + '/'
        run_number = int(sys.argv[3])
        if(config_file_name[-7:] == '.config'):
            params_dictionary = ReduceDictionary.LoadDictionary(config_file_name)
            createMTZFile(params_dictionary, out_dir, run_number)
        elif(config_file_name[-4:] == '.pkl'):
            params_dictionary = parsePickledParameters(config_file_name)
            createMTZFile(params_dictionary, out_dir, run_number,
                          run_numbers_to_process = params_dictionary['run_nums'])
        else:
            raise ValueError("The config file name must end with \".config\" or \".pkl\"")
