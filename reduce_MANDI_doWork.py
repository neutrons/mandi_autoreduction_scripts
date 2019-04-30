import matplotlib
matplotlib.use("agg")
import sys
import os
import glob
import subprocess
import re
from mantid.simpleapi import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('/SNS/snfs1/instruments/MANDI/shared/autoreduce/')
import ReduceDictionary


def get_colorscale_minimum(arr):
    x = arr[np.isfinite(arr)]
    x = x[x > 0]
    xc = x[np.argsort(x)][len(x) * 0.02]  # ignore the bottom 2%
    return xc


def makeInstrumentViewPlot(filename, outputdir):
    # instrument view plot
    event_ws = LoadEventNexus(filename)
    wh = Integration(InputWorkspace=event_ws)
    t = PreprocessDetectorsToMD(InputWorkspace=wh)
    xyz = np.array(t.column(0))  # slow
    l2 = np.array(t.column(1))
    xyz = xyz * l2[:, np.newaxis]
    y = xyz[:, 1]
    th = np.degrees(-np.arctan2(xyz[:, 0], xyz[:, 2]))
    intensity = wh.extractY()
    nbanks = th.shape[0] / (256 * 256)
    omega = wh.getRun()['omega'].getStatistics().mean
    phi = wh.getRun()['phi'].getStatistics().mean
    fig, ax = plt.subplots()
    for i in range(int(nbanks)):
        ax.pcolormesh(th[i * 256 * 256:(i + 1) * 256 * 256].reshape(256, 256),
                      y[i * 256 * 256:(i + 1) * 256 * 256].reshape(256, 256),
                      intensity[i * 256 * 256:(i + 1) * 256 * 256].reshape(256, 256),
                      vmin=0, vmax=intensity.max())
    ax.text(-130, -.35, 'Omega = {0:.2f}, Phi = {1:.2f}'.format(omega, phi))
    ax.set_xlabel('In-plane angle')
    ax.set_ylabel('Vertical distance')
    ax.set_title(wh.getTitle())

    run_number = wh.getRunNumber()
    img_filename = os.path.join(outputdir, 'MANDI_{0}_IV.png'.format(run_number))
    fig.savefig(img_filename)


def publish_plots(run_number):
    from postprocessing.publish_plot import publish_plot
    plot_html_inst = '<div><img style="max-width:90%" src="/static/web_monitor/images/MANDI_{0}_IV.png" alt="Instrument view"></div>\n'.format(run_number)  # noqa: E501
    publish_plot("MANDI", run_number, files={'file': plot_html_inst})


def createMTZFile(d, out_dir, run_number):
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
    pbpDir = d['pbpDir']
    laueLibDir = d['laueLibDir']
    laueNormBin = d['laueNormBin']
    tolerance = float(d['tolerance'])
    force_lattice_parameters = bool(d['force_lattice_parameters'])
    laue_directory = out_dir + 'laue/'

    # Create the combined workspaces and a pandas dataframe that we can use to filter bad fits.
    outputFilenameTemplate = out_dir + '%s_ws_%i_mandi_autoreduced.%s'
    combinedFilenameTemplate = out_dir + '%s_combined.integrate'
    runNumbersProcessed = []
    dfList = []
    for rn in range(first_run_number, run_number + 1):
        print('createMTZ - starting run %i' % rn)
        paramsFileName = outputFilenameTemplate % ('params', rn, 'nxs')
        peaksFileName = outputFilenameTemplate % ('peaks', rn, 'integrate')
        peaksPFFileName = outputFilenameTemplate % ('peaks_profileFitted', rn, 'integrate')
        matFileName = outputFilenameTemplate % ('UB', rn, 'mat')
        if (os.path.isfile(paramsFileName) and os.path.isfile(peaksFileName) and
                os.path.isfile(peaksPFFileName) and os.path.isfile(matFileName)):
            logger.information('Including run number {0:d}'.format(rn))
            runNumbersProcessed.append(rn)
            LoadIsawPeaks(Filename=peaksFileName, OutputWorkspace='peaks_ws')
            LoadIsawPeaks(Filename=peaksPFFileName, OutputWorkspace='peaks_ws_profile')
            Load(Filename=paramsFileName, OutputWorkspace='params_ws')

            dfTWS = pd.DataFrame(mtd['peaks_ws'].toDict())
            dfTParams = pd.DataFrame(mtd['params_ws'].toDict())
            dfT = pd.merge(dfTWS, dfTParams, left_on='PeakNumber', right_on='peakNumber', how='outer')
            dfT['theta'] = dfT['QLab'].apply(lambda x: np.arctan2(x[2], np.hypot(x[0], x[1])))
            dfT['phi'] = dfT['QLab'].apply(lambda x: np.arctan2(x[1], x[0]))
            dfList.append(dfT)

        if len(runNumbersProcessed) == 1:  # First peak we've added
            SaveIsawPeaks(InputWorkspace='peaks_ws', Filename=combinedFilenameTemplate % ('peaks'), AppendFile=False)
            SaveIsawPeaks(InputWorkspace='peaks_ws_profile', Filename=combinedFilenameTemplate % ('peaks_profileFitted'), AppendFile=False)
        else:  # Append the current workspaces
            SaveIsawPeaks(InputWorkspace='peaks_ws', Filename=combinedFilenameTemplate % ('peaks'), AppendFile=True)
            SaveIsawPeaks(InputWorkspace='peaks_ws_profile', Filename=combinedFilenameTemplate % ('peaks_profileFitted'), AppendFile=True)

        print('createMTZ - finished run %i' % rn)
    if (len(dfList) > 0):
        df = pd.concat(dfList)
        df = df.reset_index()
        pwsSPH = LoadIsawPeaks(Filename=combinedFilenameTemplate % ('peaks'), OutputWorkspace='pwsSPH')
        pwsPF = LoadIsawPeaks(Filename=combinedFilenameTemplate % ('peaks'), OutputWorkspace='pwsPF')
    else:
        logger.error('No runs to be added to create the mtz file!  Exiting!')
        sys.exit()

    # Create graphs which can be displayed on monitor
    gIDX = (df['chiSq'] < 50) & (df['chiSq3d'] < 10)
    plt.figure(1, figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df[gIDX]['Intens'], df[gIDX]['Intens3d'], '.', ms=2)
    plt.plot([1, df[gIDX]['Intens'].max()], [1, df[gIDX]['Intens'].max()], alpha=0.8)
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
    plt.plot(df[gIDX]['Energy'], df[gIDX]['Alpha'], '.', ms=1.5, label='Alpha', alpha=0.2)
    plt.plot(df[gIDX]['Energy'], df[gIDX]['Beta'], '.', ms=1.5, label='Beta', alpha=0.2)
    plt.plot(df[gIDX]['Energy'], df[gIDX]['R'], '.', ms=1.5, label='R', alpha=0.2)
    plt.legend(loc='best')
    plt.xlabel('Energy (meV)')
    plt.ylabel('Value')
    plt.title('All Peaks')
    plt.subplot(1, 2, 2)
    plt.plot(df[strongIDX]['Energy'], df[strongIDX]['Alpha'], '.', ms=1.5, label='Alpha', alpha=0.2)
    plt.plot(df[strongIDX]['Energy'], df[strongIDX]['Beta'], '.', ms=1.5, label='Beta', alpha=0.2)
    plt.plot(df[strongIDX]['Energy'], df[strongIDX]['R'], '.', ms=1.5, label='R', alpha=0.2)
    plt.legend(loc='best')
    plt.xlabel('Energy (meV)')
    plt.ylabel('Value')
    plt.title('Strong Peaks')
    plt.savefig(out_dir + '{0:d}_fig2.png'.format(run_number))

    plt.figure(3, figsize=(12, 4))
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(df[gIDX]['theta'], df[gIDX]['SigX'], '.', ms=1, alpha=0.3)
    plt.xlabel('Theta (along scattering direction) (rad)')
    plt.ylabel('Sigma Scattering (rad)')
    plt.title('All Peaks')
    plt.subplot(1, 2, 2)
    plt.plot(df[strongIDX]['theta'], df[strongIDX]['SigX'], '.', ms=1, alpha=0.3)
    plt.xlabel('Theta (along scattering direction) (rad)')
    plt.ylabel('Sigma Scattering (rad)')
    plt.title('Strong Peaks')
    plt.savefig(out_dir + '{0:d}_fig3.png'.format(run_number))

    plt.figure(4, figsize=(12, 4))
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(df[gIDX]['phi'], df[gIDX]['SigY'], '.', ms=1, alpha=0.2)
    plt.xlabel('Phi_azimuthal (rad)')
    plt.ylabel('Sigma azimuthal (rad)')
    plt.title('All Peaks')
    plt.subplot(1, 2, 2)
    plt.plot(df[strongIDX]['phi'], df[strongIDX]['SigY'], '.', ms=1, alpha=0.2)
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
    LoadIsawUB(InputWorkspace=pwsPF, Filename=outputFilenameTemplate % ('UB', runNumbersProcessed[gIDX], 'mat'))
    LoadIsawUB(InputWorkspace=pwsSPH, Filename=outputFilenameTemplate % ('UB', runNumbersProcessed[gIDX], 'mat'))
    numIndexed = IndexPeaks(pwsPF, tolerance=tolerance)[0]
    numIndexed = IndexPeaks(pwsSPH, tolerance=tolerance)[0]
    print('There are {0:d} peaks total.'
          ' The UB matrix from run {1:d} will index '
          '{2:d} of them ({3:4.2f} percent).  '
          'Using this file.'.format(pwsPF.getNumberPeaks(), runNumbersProcessed[gIDX],
                                    numPeaksIndexed[gIDX],
                                    100. * numPeaksIndexed[gIDX] / pwsPF.getNumberPeaks()))

    # It is helpful to use force_lattice_parameters if the UB files being loaded are the Niggli cell.
    if force_lattice_parameters:
        print('Reindexing peaks in new coordinate system.  This may take serveral minutes.')
        FindUBUsingLatticeParameters(PeaksWorkspace=pwsPF, a=a, b=b, c=c,
                                     alpha=alpha, beta=beta, gamma=gamma,
                                     NumInitial=50, Tolerance=tolerance, Iterations=1000)
        FindUBUsingLatticeParameters(PeaksWorkspace=pwsSPH, a=a, b=b, c=c,
                                     alpha=alpha, beta=beta, gamma=gamma,
                                     NumInitial=50, Tolerance=tolerance, Iterations=1000)
        numIndexed = IndexPeaks(PeaksWorkspace=pwsPF)[0]
        numIndexed = IndexPeaks(PeaksWorkspace=pwsSPH)[0]
        lattice = pwsPF.sample().getOrientedLattice()
        print('New lattice:')
        print(lattice)
        print('Indexes {0:d} of {1:d} peaks'.format(numIndexed, pwsPF.getNumberPeaks()))

    df['h_reindexed'] = pwsPF.column('h')
    df['k_reindexed'] = pwsPF.column('k')
    df['l_reindexed'] = pwsPF.column('l')

    # Write our mtz files
    goodIDX = (df['chiSq'] < 50.0) & (df['chiSq3d'] < 10)
    edgeIDX = (df['Row'] <= lauenorm_edge_pixels) | (df['Row'] >= 255 - lauenorm_edge_pixels) | (
        df['Col'] <= lauenorm_edge_pixels) | (df['Col'] >= 255 - lauenorm_edge_pixels)
    print('Rejecting {0} peaks for bad fits and {1} peaks for being on the edge'.format(np.sum(~goodIDX), np.sum(edgeIDX)))
    goodIDX = goodIDX & ~edgeIDX

    ws = CloneWorkspace(InputWorkspace=pwsPF, OutputWorkspace='ws')
    ws2 = CloneWorkspace(InputWorkspace=pwsSPH, OutputWorkspace='ws2')
    for i in range(len(df)):
        if goodIDX[i]:
            ws.getPeak(i).setIntensity(df.iloc[i]['Intens3d'])
            ws.getPeak(i).setSigmaIntensity(df.iloc[i]['SigInt3d'])
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
    SaveLauenorm(InputWorkspace=ws, Filename=laue_directory + 'laueNorm',
                 ScalePeaks=lauenorm_scale_peaks, MinDSpacing=lauenorm_min_d, MinWavelength=lauenorm_min_wl,
                 MaxWavelength=lauenorm_max_wl, SortFilesBy='RunNumber', MinIsigI=lauenorm_min_isi, MinIntensity=lauenorm_mini)
    print('Wrote laueNorm input files to %s' % (laue_directory))

    comFilename = laue_directory + 'lnorm.com'
    datFilename = laue_directory + 'lnorm.dat'
    datFilenameMerged = laue_directory + 'lnorm_merged.dat'
    numRuns = len(np.unique(ws.column('RunNumber')))
    lattice = pwsPF.sample().getOrientedLattice()

    # unmerged .dat file
    with open(datFilename, 'w') as f:
        f.write('5s70aMaNDi3\n')
        f.write('%2.2f %2.2f %2.2f %i %i %i\n' % (lattice.a(), lattice.b(), lattice.c(),
                                                  np.round(lattice.alpha()), np.round(lattice.beta()), np.round(lattice.gamma())))
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
        f.write('%2.2f %2.2f %2.2f %i %i %i\n' % (lattice.a(), lattice.b(), lattice.c(),
                                                  np.round(lattice.alpha()), np.round(lattice.beta()), np.round(lattice.gamma())))
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
        f.write('source /SNS/snfs1/instruments/MANDI/shared/laue3/laue/laue.setup-sh\n')
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
        f.write('time %s < %slnorm.dat > %slnorms70aMaNDi.log\n' % (laueNormBin, laue_directory, laue_directory))
        f.write('HKLOUT=$cwd/%s_merged.mtz\n' % mtz_name)
        f.write('export HKLOUT\n')
        f.write('time %s < %slnorm_merged.dat > %slnorms70aMaNDi_merged.log\n' % (laueNormBin, laue_directory, laue_directory))
    os.chmod(comFilename, 0775)
    print('Wrote lauenorm executable to %s' % comFilename)
    print('Running laueNorm...')
    mtd.clear()
    subprocess.Popen(comFilename, cwd=os.path.dirname(os.path.realpath(comFilename)))


def doIntegration(d, nxsFilename, out_dir, run_number):
    rA = d['peak_radius']
    rB = d['bkg_inner_radius']
    rC = d['bkg_outer_radius']
    min_d = d['min_d']
    max_d = d['max_d']
    tolerance = d['tolerance']
    moderatorFile = d['moderator_file']
    predictPeaks = d['integrate_predicted_peaks']
    min_pred_dspacing = d['min_pred_dspacing']
    max_pred_dspacing = d['max_pred_dspacing']
    min_pred_wl = d['min_pred_wl']
    max_pred_wl = d['max_pred_wl']
    StrongPeaksParamsFile = d['strong_peaks_params_file']
    IntensityCutoff = d['intensity_cutoff']
    EdgeCutoff = d['edge_cutoff']
    FracStop = d['frac_stop']
    MinpplFrac = d['min_ppl_frac']
    MaxpplFrac = d['max_ppl_frac']
    DQMax = d['dq_max']
    a = d['unitcell_a']
    b = d['unitcell_b']
    c = d['unitcell_c']
    alpha = d['unitcell_alpha']
    beta = d['unitcell_beta']
    gamma = d['unitcell_gamma']
    num_peaks_to_find = d['num_peaks_to_find']
    DetCalFile = d['calibration_file_1']
    outputFilenameTemplate = out_dir + '%s_ws_%i_mandi_autoreduced.%s'  # String with output file format.  %s will be replaced by file
    try:
        event_ws = Load(Filename=nxsFilename, OutputWorkspace='event_ws')
        if DetCalFile is not None:
            print('Loading DetCal file %s' % DetCalFile)
            LoadIsawDetCal(InputWorkspace=event_ws, Filename=DetCalFile)
        ConvertToMD(InputWorkspace='event_ws', QDimensions='Q3D', dEAnalysisMode='Elastic',
                    Q3DFrames='Q_lab', OutputWorkspace='MDdata', MinValues='-5,-5,-5',
                    MaxValues='5,5,5', MaxRecursionDepth=10)
        peaks_ws = FindPeaksMD(InputWorkspace='MDdata', MaxPeaks=num_peaks_to_find, DensityThresholdFactor=500, OutputWorkspace='peaks_ws')
        try:
            FindUBUsingFFT(PeaksWorkspace='peaks_ws', MinD=min_d, MaxD=max_d, Tolerance=tolerance, Iterations=10, DegreesPerStep=1.0)
        except:
            FindUBUsingLatticeParameters(PeaksWorkspace='peaks_ws', a=a, b=b, c=c, alpha=alpha, beta=beta,
                                         gamma=gamma, NumInitial=50, Tolerance=tolerance, Iterations=1000)
        IndexPeaks(PeaksWorkspace='peaks_ws')
        mtd.remove('event_ws')  # Free up memory
        if predictPeaks:
            print("PREDICTING peaks to integrate....")
            peaks_ws = PredictPeaks(InputWorkspace=peaks_ws,
                                    WavelengthMin=min_pred_wl, WavelengthMax=max_pred_wl,
                                    MinDSpacing=min_pred_dspacing, MaxDSpacing=max_pred_dspacing,
                                    ReflectionCondition='Primitive')

        if np.max([a, b, c] > 150):
            SetInstrumentParameter(Workspace='peaks_ws', ParameterName='fracHKL', ParameterType='Number', Value='0.4')

        IntegratePeaksMD(InputWorkspace='MDdata', PeakRadius=rA, BackgroundInnerRadius=rB, BackgroundOuterRadius=rC,
                         PeaksWorkspace='peaks_ws', OutputWorkspace='peaks_ws', CylinderLength=0.4, PercentBackground=20,
                         ProfileFunction='IkedaCarpenterPV')

        IntegratePeaksProfileFitting(OutputPeaksWorkspace='peaks_ws_out', OutputParamsWorkspace='params_ws', InputWorkspace='MDdata',
                                     PeaksWorkspace='peaks_ws', ModeratorCoefficientsFile=moderatorFile, DQMax=DQMax,
                                     MinpplFrac=MinpplFrac, MaxpplFrac=MaxpplFrac, FracStop=FracStop, EdgeCutoff=EdgeCutoff,
                                     IntensityCutoff=IntensityCutoff, StrongPeakParamsFile=StrongPeaksParamsFile)

        paramsFileName = outputFilenameTemplate % ('params', run_number, 'nxs')
        peaksFileName = outputFilenameTemplate % ('peaks', run_number, 'integrate')
        peaksPFFileName = outputFilenameTemplate % ('peaks_profileFitted', run_number, 'integrate')
        matFileName = outputFilenameTemplate % ('UB', run_number, 'mat')
        if os.path.isfile(paramsFileName):
            os.remove(paramsFileName)
        if os.path.isfile(peaksFileName):
            os.remove(peaksFileName)
        if os.path.isfile(peaksPFFileName):
            os.remove(peaksPFFileName)
        if os.path.isfile(peaksPFFileName):
            os.remove(matFileName)

        SaveNexus(InputWorkspace='params_ws', Filename=paramsFileName)
        SaveIsawPeaks(InputWorkspace='peaks_ws', Filename=peaksFileName)
        SaveIsawPeaks(InputWorkspace='peaks_ws_out', Filename=peaksPFFileName)
        SaveIsawUB(InputWorkspace='peaks_ws', Filename=matFileName)
        mtd.clear()
    except:
        raise
        # raise UserWarning('ERROR WITH RUN %i'%run_number)


def do_reduction(filename, outputdir):
    # Do profile fitting
    run_number = int(re.search('MANDI_(\d+).nxs.h5', filename).group(1))
    config_filename = outputdir + 'mandi_autoreduce.config'
    config_file_list = glob.glob(config_filename)
    if len(config_file_list) == 1:
        d = ReduceDictionary.LoadDictionary(config_filename)
        doIntegration(d, filename, outputdir, run_number)
        mtd.clear()  # Free some memory

        # Create the mtz
        print('Moving on to createMTZFile')
        createMTZFile(d, outdir, run_number)
        mtd.clear()  # Free some memory

        # Generate the instrument plot
        makeInstrumentViewPlot(filename, outdir)
        mtd.clear()

        try:
            publish_plots(run_number)
        except:
            print('Cannot publish plots. Maybe this was not run using autoreduction?')

    else:
        raise UserWarning("Config file {0} does not exist.  Cannot do profile fitting.".format(config_filename))


if __name__ == "__main__":
    np.seterr("ignore")  # ignore division by 0 warning in plots
    # check number of arguments
    if (len(sys.argv) != 3):
        logger.error("autoreduction code requires a filename and an output directory")
        sys.exit()
    if not(os.path.isfile(sys.argv[1])):
        logger.error("data file " + sys.argv[1] + " not found")
        sys.exit()
    else:
        filename = sys.argv[1]
        outdir = sys.argv[2]
        do_reduction(filename, outdir)
