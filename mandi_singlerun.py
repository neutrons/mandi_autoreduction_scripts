from __future__ import (absolute_import, division, print_function)
import os
import sys
import ReduceDictionary
import matplotlib
import numpy as np
matplotlib.use('Agg')  # autoanalyzer1 fails if we dont set this
sys.path.insert(0, "/opt/mantidnightly/bin")
from mantid.simpleapi import *  # noqa: F402, F403
from mantid.api import *  # noqa: F402, F403


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
    outputFilenameTemplate = out_dir + '%s_ws_%i_mandi_autoreduced.%s'
    try:
        event_ws = Load(Filename=nxsFilename, OutputWorkspace='event_ws')
        #filter low (or zero) power pulses
        event_ws = FilterByLogValue(InputWorkspace=event_ws, LogName='proton_charge', MinimumValue=2e6)
        if DetCalFile is not None:
            print('Loading DetCal file %s' % DetCalFile)
            LoadIsawDetCal(InputWorkspace=event_ws, Filename=DetCalFile)
        ConvertToMD(InputWorkspace='event_ws', QDimensions='Q3D',
                    dEAnalysisMode='Elastic', Q3DFrames='Q_lab',
                    OutputWorkspace='MDdata', MinValues='-5,-5,-5',
                    MaxValues='5,5,5', MaxRecursionDepth=10)
        peaks_ws = FindPeaksMD(InputWorkspace='MDdata',
                               MaxPeaks=num_peaks_to_find,
                               DensityThresholdFactor=500,
                               OutputWorkspace='peaks_ws')

        if np.max([a, b, c] > 150):
            SetInstrumentParameter(Workspace='peaks_ws',
                                   ParameterName='fracHKL',
                                   ParameterType='Number',
                                   Value='0.4')
        try:
            FindUBUsingFFT(PeaksWorkspace='peaks_ws', MinD=min_d, MaxD=max_d,
                           Tolerance=tolerance, Iterations=10,
                           DegreesPerStep=1.0)
        except:
            FindUBUsingLatticeParameters(PeaksWorkspace='peaks_ws',
                                         a=a, b=b, c=c,
                                         alpha=alpha, beta=beta, gamma=gamma,
                                         NumInitial=50, Tolerance=tolerance,
                                         Iterations=1000)
        IndexPeaks(PeaksWorkspace='peaks_ws')
        mtd.remove('event_ws')  # Free up memory
        if predictPeaks:
            print("PREDICTING peaks to integrate....")
            peaks_ws = PredictPeaks(InputWorkspace=peaks_ws,
                                    WavelengthMin=min_pred_wl,
                                    WavelengthMax=max_pred_wl,
                                    MinDSpacing=min_pred_dspacing,
                                    MaxDSpacing=max_pred_dspacing,
                                    ReflectionCondition='Primitive')

        IntegratePeaksMD(InputWorkspace='MDdata', PeakRadius=rA,
                         BackgroundInnerRadius=rB, BackgroundOuterRadius=rC,
                         PeaksWorkspace='peaks_ws', OutputWorkspace='peaks_ws',
                         CylinderLength=0.4, PercentBackground=20,
                         ProfileFunction='IkedaCarpenterPV')

        IntegratePeaksProfileFitting(OutputPeaksWorkspace='peaks_ws_out',
                                     OutputParamsWorkspace='params_ws',
                                     InputWorkspace='MDdata',
                                     PeaksWorkspace='peaks_ws',
                                     ModeratorCoefficientsFile=moderatorFile,
                                     DQMax=DQMax, MinpplFrac=MinpplFrac,
                                     MaxpplFrac=MaxpplFrac, FracStop=FracStop,
                                     EdgeCutoff=EdgeCutoff,
                                     IntensityCutoff=IntensityCutoff,
                                     StrongPeakParamsFile=StrongPeaksParamsFile)  # noqa: E501

        paramsFileName = outputFilenameTemplate % ('params', run_number, 'nxs')
        peaksFileName = outputFilenameTemplate % ('peaks',
                                                  run_number, 'integrate')
        peaksPFFileName = outputFilenameTemplate % ('peaks_profileFitted',
                                                    run_number, 'integrate')
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


if __name__ == '__main__':
    # arg1=script name, arg2=config filename, arg3=nxs filename,
    # arg4=outputdir, arg5=run_number
    if (len(sys.argv) != 5):
        logger.error('{0} must take 4 arguments!'.format(sys.argv[0]))
        sys.exit()
    if not(os.path.isfile(sys.argv[1])):
        logger.error("config file " + sys.argv[1] + " not found")
        sys.exit()
    if not(os.path.isfile(sys.argv[2])):
        logger.error("data file " + sys.argv[2] + " not found")
        sys.exit()
    else:
        config_file_name = str(sys.argv[1])
        nxs_name = str(sys.argv[2])
        out_dir = str(sys.argv[3])
        run_number = int(sys.argv[4])
        params_dictionary = ReduceDictionary.LoadDictionary(config_file_name)
        doIntegration(params_dictionary, nxs_name, out_dir, run_number)
