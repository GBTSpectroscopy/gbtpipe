#from sdpy import makecube
import numpy as np
import glob
from astropy.io import fits
import astropy.wcs as wcs
import itertools
from scipy.special import j1
import pdb
import numpy.fft as fft
import astropy.utils.console as console
import astropy.units as u
import astropy.constants as con
import numpy.polynomial.legendre as legendre
import warnings
from .Baseline import *
import os
from spectral_cube import SpectralCube
from radio_beam import Beam
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
from . import __version__

# Restriction: Mask must be in the same spectral space as the resulting
# gridded cube.

def buildMaskLookup(filename):
    maskcube = SpectralCube.read(filename)
    wcs = maskcube.wcs
    mask = np.array(maskcube.filled_data[:], dtype=np.bool)
    spatial_mask = np.any(mask, axis=0)        
    nuinterp = interp1d(maskcube.spectral_axis.to(u.Hz).value,
                        np.arange(maskcube.shape[0]),
                        bounds_error=False,
                        fill_value='extrapolate')
    def maskLookup(ra, dec, freq):
        xx, yy, zz = wcs.wcs_world2pix(ra, dec, freq, 0)
        if (0 <= xx[0] < spatial_mask.shape[1] and
            0 <= yy[0] < spatial_mask.shape[0]):
            emission = spatial_mask[int(yy[0]), int(xx[0])]
        else:
            emission = False
        if emission:
            zz = np.array(nuinterp(freq), dtype=np.int)
            zz[zz < 0] = 0
            zz[zz >= mask.shape[0]] = mask.shape[0] - 1
            return(mask[zz, yy.astype(np.int), xx.astype(np.int)])
        else:
            return(np.zeros_like(freq, dtype=np.bool))
    return(maskLookup)

def drawTimeSeriesPlot(data, filename='TimeSeriesPlot',
                       suffix='png', outdir=None, plotsubdir='',
                       flags=None):
    if outdir is None:
        outdir = os.getcwd()
    if not os.access(outdir, os.W_OK):
        os.mkdir(outdir)
    if not os.access(outdir +'/' + plotsubdir, os.W_OK):
        os.mkdir(outdir + '/' + plotsubdir)

    plt.switch_backend('agg')
    # fix for subdirectories.
    vmin=np.nanpercentile(data,15)
    vmed=np.nanpercentile(data,50)
    vmax=np.nanpercentile(data,85)
    fig = plt.figure(figsize=(8.0,6.5))
    ax = fig.add_subplot(111)
    if flags is not None:
        flagmask = 1-flags[np.newaxis].T * np.ones((1, data.shape[1]))
    else:
        flagmask = 1.0
    im = ax.imshow(data * flagmask,
                    interpolation='nearest',
                    cmap='PuOr', vmin=(4*vmin-3*vmed),
                    vmax=4*vmax-3*vmed)
    ax.set_xlabel('Channel')
    # ax.set_title((filename.split('/'))[-1])
    ax.set_ylabel('Scan')
    cb = fig.colorbar(im)
    cb.set_label('Intensity (K)')
    thisroot = (filename.split('/'))[-1]
    plt.savefig(outdir + '/' + plotsubdir +
                '/' + thisroot.replace('fits', suffix))
    plt.close()
    plt.clf()


def freqShiftValue(freqIn, vshift, convention='RADIO'):
    cms = 299792458.
    if convention.upper() in 'OPTICAL':
        return freqIn / (1.0 + vshift / cms)
    if convention.upper() in 'TRUE':
        return freqIn * ((cms + vshift) / (cms - vshift))**0.5
    if convention.upper() in 'RADIO':
        return freqIn * (1.0 - vshift / cms)


def channelShift(x, ChanShift):
    # Shift a spectrum by a set number of channels.
    ftx = np.fft.fft(x)
    m = np.fft.fftfreq(len(x))
    phase = np.exp(2 * np.pi * m * 1j * ChanShift)
    x2 = np.real(np.fft.ifft(ftx * phase))
    return(x2)


def VframeInterpolator(scan):
    # Find cases where the scan number is
    startidx = scan['PROCSEQN']!=np.roll(scan['PROCSEQN'],1)
    scanstarts = scan[startidx]
    indices = np.arange(len(scan))
    startindices = indices[startidx]
    scannum = scanstarts['PROCSEQN']
    vfs = scanstarts['VFRAME']

    odds = (scannum % 2) == 1
    evens = (scannum % 2) == 0

    coeff_odds,_,_,_ = np.linalg.lstsq(\
        np.c_[startindices[odds]*1.0,
              np.ones_like(startindices[odds])],
        vfs[odds])

    coeff_evens,_,_,_ = np.linalg.lstsq(\
        np.c_[startindices[evens]*1.0,
              np.ones_like(startindices[evens])],
        vfs[evens])

    vfit = np.zeros(len(scan))+np.nan

    for thisone, singlescan in enumerate(scan):
        startv = vfs[scannum == singlescan['PROCSEQN']]
        startt = startindices[scannum == singlescan['PROCSEQN']]
        if singlescan['PROCSEQN'] % 2 == 0:
            endv = coeff_odds[1] + coeff_odds[0] * (startt+94)
        if singlescan['PROCSEQN'] % 2 == 1:
            endv = coeff_evens[1] + coeff_evens[0] * (startt+94)

        endt = startt+94
        try:
            vfit[thisone] = (thisone - startt) * \
                (endv - startv)/94 + startv
        except:
            pass
    return(vfit)


def preprocess(filename,
               startChannel=None,
               endChannel=None,
               doBaseline=True,
               baselineRegion=None,
               blorder=1,
               OnlineDoppler=True,
               flagRMS=True,
               rmsThresh=1.25,
               flagRipple=True,
               rippleThresh=2,
               plotTimeSeries=False,
               spikeThresh=10,
               flagSpike=True,
               windowStrategy='simple',
               maskfile=None,
               edgefraction=0.05,
               restFrequences=None,
               gainDict=None,
               outdir=None,
               plotsubdir='',
               robust=False,
               **kwargs):

    # Constants block
    sqrt2 = np.sqrt(2)
    mad2rms = 1.4826
    prefac = mad2rms / sqrt2
    c = 299792458.
    ####################

    s = fits.getdata(filename)

    nData = len(s[0]['DATA'])

    if startChannel is None:
        startChannel = int(edgefraction * nData)
    if endChannel is None:
        endChannel = int((1 - edgefraction) * nData)

    nChannel = endChannel - startChannel
    if baselineRegion is None:
        baselineRegion = [slice(startChannel, endChannel, 1)]
    
    if outdir is None:
        outdir = os.getcwd()

    crval3 = s[0]['RESTFREQ'] * (1 - s[0]['VELOCITY'] / c)
    crpix3 = s[0]['CRPIX1'] - startChannel
    ctype3 = s[0]['CTYPE1']
    cdelt3 = s[0]['CDELT1'] 
    spectral_axis = (np.arange(nData)
                     + 1 - crpix3) * cdelt3 + crval3
    
    if not OnlineDoppler:
        vframe_list = VframeInterpolator(s)
    else:
        vframe_list = s['VFRAME']

    # BEFORE PLOT
    if plotTimeSeries:
        drawTimeSeriesPlot(s['DATA'], filename=plotsubdir + filename)
    
    if windowStrategy == 'cubemask':
        maskLookup = buildMaskLookup(maskfile)

    outscans = []
    outwts = []
    tsyslist = []
    flagct = 0

    for idx, (spectrum, vframe) in enumerate(zip(s, vframe_list)):
        if spectrum['OBJECT'] == 'VANE' or spectrum['OBJECT'] == 'SKY':
            continue

        specData = spectrum['DATA']
        DeltaNu = freqShiftValue(spectrum['CRVAL1'],
                                    -vframe) - crval3
        DeltaChan = DeltaNu / cdelt3
        specData = channelShift(specData, -DeltaChan)
        baselineMask = np.zeros_like(specData, dtype=np.bool)
        noise = None
        if flagSpike:
            jumps = (specData - np.roll(specData, -1))
            noise = mad1d(jumps) * 2**(-0.5)
            spikemask = (np.abs(jumps) < spikeThresh * noise)
            spikemask = spikemask * np.roll(spikemask, 1)
            specData[~spikemask] = 0.0
        else:
            spikemask = np.ones_like(specData, dtype=np.bool)

        # This part takes the TOPOCENTRIC frequency that is at
        # CRPIX1 (i.e., CRVAL1) and calculates the what frequency
        # that would have in the LSRK frame with freqShiftValue.
        # This then compares to the desired frequency CRVAL3.

        if windowStrategy == 'simple':
            baselineIndex = simpleWindow(spectrum,
                                         edgefraction=edgefraction,
                                         **kwargs)
            for r in baselineIndex:
                baselineMask[r] = True

        if windowStrategy == 'cubemask':
            baselineMask[:] = True
            baselineMask[0:int(edgefraction * nData)] = False
            baselineMask[int((1 - edgefraction) * nData):] = False
            thismask = maskLookup((spectrum['CRVAL2'] 
                                   * np.ones_like(spectral_axis)),
                                  (spectrum['CRVAl3']
                                   * np.ones_like(spectral_axis)),
                                  spectral_axis)
            baselineMask[np.squeeze(thismask.astype(np.bool))] = False
        if doBaseline & np.all(np.isfinite(specData[baselineMask])):
            if robust:
                specData = robustBaseline(specData, blorder=blorder,
                                          baselineIndex=baselineMask,
                                          noiserms=noise)
            else:
                specData = baselineSpectrum(specData, order=blorder,
                                            baselineIndex=baselineMask)
        if gainDict:
            try:
                feedwt = 1.0/gainDict[(str(spectrum['FDNUM']).strip(),
                                        str(spectrum['PLNUM']).strip())]
            except KeyError:
                continue
        else:
            feedwt = 1.0

        tsys = spectrum['TSYS']
        outslice = (specData)[startChannel:endChannel]
        
        if flagRMS:
            radiometer_rms = tsys / np.sqrt(np.abs(spectrum['CDELT1']) *
                                            spectrum['EXPOSURE'])
            scan_rms = prefac * np.median(np.abs(outslice[0:-2] -
                                                    outslice[2:]))

            if scan_rms > rmsThresh * radiometer_rms:
                tsys = 0 # Blank spectrum
                
        if flagRipple:
            scan_rms = prefac * np.median(np.abs(outslice[0:-2] -
                                                    outslice[2:]))
            ripple = prefac * sqrt2 * np.median(np.abs(outslice))

            if ripple > rippleThresh * scan_rms:
                tsys = 0 # Blank spectrum

        if tsys == 0:
            flagct +=1

        spectrum_wt = ((np.isfinite(outslice)
                        * spikemask[startChannel:
                                    endChannel]).astype(np.float)
                        * feedwt)
        outslice = np.nan_to_num(outslice)
        outscans += [outslice]
        outwts += [spectrum_wt]
        tsyslist += [tsys]

    print ("Percentage of flagged scans: {0:4.2f}".format(
           100*flagct/float(idx)))

    # AFTER PLOT
    if plotTimeSeries:
        drawTimeSeriesPlot(np.array(outscans),
                           filename=plotsubdir + filename,
                           suffix='flagged.png',
                           flags=(np.array(tsyslist) == 0))

    return(s, np.array(outscans), np.array(outwts), np.array(tsyslist))
