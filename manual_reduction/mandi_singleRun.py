from __future__ import (absolute_import, division, print_function)
import os
import sys
import pickle
from mantid.simpleapi import *


def doIntegration(d):
    runNumber = d['runNumber']
    nxsFormat = d['eventFilesFormat']
    rA = d['rA']
    rB = d['rB']
    rC = d['rC']
    minD = d['minD']
    maxD = d['maxD']
    tol = d['tol']
    moderatorFile = d['ModeratorFile']
    outDir = d['outDir']
    predictPeaks = d['predictPeaks']
    min_pred_dspacing = d['min_pred_dspacing']
    max_pred_dspacing = d['max_pred_dspacing']
    min_pred_wl = d['min_pred_wl']
    max_pred_wl = d['max_pred_wl']
    StrongPeaksParamsFile = d['StrongPeaksParamsFile']
    IntensityCutoff = d['IntensityCutoff']
    EdgeCutoff = d['EdgeCutoff']
    FracStop = d['FracStop']
    MinpplFrac = d['MinpplFrac']
    MaxpplFrac = d['MaxpplFrac']
    DQMax = d['DQMax']
    a = d['a']
    b = d['b']
    c = d['c']
    alpha = d['alpha']
    beta = d['beta']
    gamma = d['gamma']
    DetCalFile = d['DetCalFile']

    outputFilenameTemplate = outDir+'%s_ws_%i_mandi_reduced.%s' #String with output file format.  %s will be

    try:
        event_ws = Load(Filename=nxsFormat%runNumber, OutputWorkspace='event_ws')
        if DetCalFile is not None:
            print('Loading DetCal file %s'%DetCalFile)
            LoadIsawDetCal(InputWorkspace=event_ws, Filename=DetCalFile)
        ConvertToMD(InputWorkspace='event_ws', QDimensions='Q3D', dEAnalysisMode='Elastic', Q3DFrames='Q_lab',
                    OutputWorkspace='MDdata', MinValues='-5,-5,-5', MaxValues='5,5,5', MaxRecursionDepth=10)
        peaks_ws = FindPeaksMD(InputWorkspace='MDdata', MaxPeaks=150, DensityThresholdFactor=500, OutputWorkspace='peaks_ws')
        try:
            FindUBUsingFFT(PeaksWorkspace='peaks_ws', MinD=minD, MaxD=maxD, Tolerance=tol, Iterations=10, DegreesPerStep=1.0)
        except:
            FindUBUsingLatticeParameters(PeaksWorkspace='peaks_ws', a=a, b=b, c=c, alpha=alpha, beta=beta,
                                         gamma=gamma, NumInitial=50, Tolerance=tol,Iterations=1000)
        IndexPeaks(PeaksWorkspace='peaks_ws')
        """
        #Set optimal crytal position. This is experimental still.
        OptimizeCrystalPlacement(PeaksWorkspace=peaks_ws, ModifiedPeaksWorkspace=peaks_ws, FitInfoTable='t2',
                                 AdjustSampleOffsets=True, OptimizeGoniometerTilt=False)
        newSampPos = [mtd['t2'].column(1)[0], mtd['t2'].column(1)[1], mtd['t2'].column(1)[2]]
        SetCrystalLocation(InputWorkspace=event_ws, OutputWorkspace=event_ws, NewX=newSampPos[0],
                           NewY=newSampPos[1], NewZ=newSampPos[2])
        MDdata = ConvertToMD(InputWorkspace='event_ws', QDimensions='Q3D', dEAnalysisMode='Elastic',
                             Q3DFrames='Q_lab', OutputWorkspace='MDdata', MinValues='-5,-5,-5',
                             MaxValues='5,5,5', MaxRecursionDepth=10)
        np.savetxt('/SNS/users/USR/Desktop/nak/%i_samppos.txt'%runNumber, newSampPos)
        """ #End set optimal xtal position

        mtd.remove('event_ws') #Free up memory
        if predictPeaks:
            print("PREDICTING peaks to integrate....")
            peaks_ws = PredictPeaks(InputWorkspace=peaks_ws,
                                    WavelengthMin=min_pred_wl, WavelengthMax=max_pred_wl,
                                    MinDSpacing=min_pred_dspacing, MaxDSpacing=max_pred_dspacing,
                                    ReflectionCondition='Primitive' )

        IntegratePeaksMD(InputWorkspace='MDdata', PeakRadius=rA, BackgroundInnerRadius=rB, BackgroundOuterRadius=rC,
                         PeaksWorkspace='peaks_ws', OutputWorkspace='peaks_ws', CylinderLength=0.4, PercentBackground=20,
                         ProfileFunction='IkedaCarpenterPV')

        #for i in range(peaks_ws.getNumberPeaks()):
        #    peaks_ws.getPeak(i).setRunNumber(runNumber)

        IntegratePeaksProfileFitting(OutputPeaksWorkspace='peaks_ws_out', OutputParamsWorkspace='params_ws', InputWorkspace='MDdata',
                                     PeaksWorkspace='peaks_ws', ModeratorCoefficientsFile=moderatorFile, DQMax=DQMax,
                                     MinpplFrac=MinpplFrac, MaxpplFrac=MaxpplFrac, FracStop=FracStop, EdgeCutoff=EdgeCutoff,
                                     IntensityCutoff=IntensityCutoff, StrongPeakParamsFile=StrongPeaksParamsFile)

        paramsFileName = outputFilenameTemplate%('params',runNumber,'nxs')
        peaksFileName = outputFilenameTemplate%('peaks',runNumber,'integrate')
        peaksPFFileName = outputFilenameTemplate%('peaks_profileFitted',runNumber,'integrate')
        matFileName = outputFilenameTemplate%('UB',runNumber,'mat')
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

    except:
        raise


if __name__ == '__main__':
    dictPath = str(sys.argv[1])
    d = pickle.load(open(dictPath, 'rb'))
    doIntegration(d)
