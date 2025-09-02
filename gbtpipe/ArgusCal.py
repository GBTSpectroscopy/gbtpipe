from __future__ import unicode_literals, print_function
from builtins import str
from .Weather import *
from .Calibration import *
from .PipeLogging import *
from .Integration import *
from .ConvenientIntegration import ConvenientIntegration
from .ObservationRows import *
from .SdFitsIO import SdFits, SdFitsIndexRowReader
from .smoothing import *
from .commandline import CommandLine
from .Gridding import griddata
from .Baseline import *
from .gbt_pipeline import *
import numpy as np
import glob
import os
# import fitsio
import copy
import warnings
import sys
import astropy.units as u
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord
from multiprocessing import Pool
from functools import partial
from scipy.interpolate import make_smoothing_spline

def makelogdir():
    # Creates log director required by default by gbtpipe
    if not os.path.exists('log'):
        os.mkdir('log')

def findfeed(cl_params, allfiles, mapscans, thisscan, feednum,
             log=None):
    """ 
    This finds identical data from another feed.
    Needed for doing the beam swap
    """
    thispol = 0
    thiswin = 0
    cl_params2 = copy.deepcopy(cl_params)
    for anotherfile in allfiles:
        cl_params2.infile = anotherfile
        sdf = SdFits()
        indexfile = sdf.nameIndexFile(anotherfile)
        row_list, _ = sdf.parseSdfitsIndex(indexfile,
                                                 mapscans=mapscans)
        feedlist = (row_list.feeds())
        if feednum in feedlist:
            rows = row_list.get(thisscan, feednum,
                                thispol, thiswin)
            pipe = MappingPipeline(cl_params,
                                   row_list,
                                   feednum,
                                   thispol,
                                   thiswin,
                                   None, outdir='.',
                                   suffix='_tmp', log=log)
            ext = rows['EXTENSION']
            rows =rows['ROW']
            columns = tuple(pipe.infile[ext].get_colnames())
            integs = ConvenientIntegration(pipe.infile[ext][columns][rows],
                                           log=log)
            pipe.infile.close()
            pipe.outfile.close()
            crap = glob.glob('*_tmp.fits')
            for thisfile in crap:
                os.remove(thisfile)
            return(integs)
    return(None)

        
def gettsys(cl_params, row_list, thisfeed, thispol, thiswin, pipe,
            opacity=True,
            weather=None, log=None):
    """
    Determine Tsys for list of map rows

    Parameters
    ----------
    cl_params : dict
        command line parameter dictionary
    row_list : Row
        Object row 
    thisfeed : int
        Feed (receptor) number
    thispol : int
        Polarization number
    thiswin : int
        Spectral window number
    pipe : dict
        Pipeline control dictionary
    weather : Weather
        Object providing weather forecasting functionality
    log : Logging
        Object providing access to log functionality.
    """

    if not weather:
        weather = Weather()  
    if not log:
        log = Logging(cl_params, 'gbtpipeline')
    cal = Calibration()

    # Assume refscan is this scan and the next scan
    thisscan = cl_params.refscans[0]

    # Pull the right raw rows from the data
    row1 = row_list.get(thisscan, thisfeed, thispol, thiswin)
    row2 = row_list.get(thisscan+1, thisfeed, thispol, thiswin)

    # Pull cal data into memory.
    ext = row1['EXTENSION']
    rows = row1['ROW']
    columns = tuple(pipe.infile[ext].get_colnames())
    integ1 = ConvenientIntegration(pipe.infile[ext][columns][rows],
                                           log=log)

    ext = row2['EXTENSION']
    rows = row2['ROW']
    columns = tuple(pipe.infile[ext].get_colnames())
    integ2 = ConvenientIntegration(pipe.infile[ext][columns][rows],
                                           log=log)

    vec1 = integ1.data['DATA']
    vec2 = integ2.data['DATA']

    if ((('Vane' in integ1.data['CALPOSITION'][0]) and
         ('Observing' in integ2.data['CALPOSITION'][0])) or 
         (('VANE' in integ1.data['OBJECT'][0]) and 
          ('SKY' in integ2.data['OBJECT'][0]))):
        # Directly do an on-off on the full data set
        onoff = np.median((vec1-vec2)/vec2)
        # Mean over time to get a vector of vaneCounts at each freq.
        vaneCounts = np.mean(vec1, axis=0)

    # Now test case where Vane data were second position
    elif ((('Vane' in integ2.data['CALPOSITION'][0]) and
           ('Observing' in integ1.data['CALPOSITION'][0])) or 
          (('VANE' in integ2.data['OBJECT'][0]) and 
           ('SKY' in integ1.data['OBJECT'][0]))):
        onoff = np.median((vec2-vec1)/vec1)
        vaneCounts = np.mean(vec2, axis=0)
    else:
        # If you are here, then you will not get data today
        log.logger.warning("No Vane scans found in putative reference scans.")
        log.logger.warning("Making a wild guess that's probably wrong...")
        onoff = np.nanmedian((vec1-vec2)/vec2)
        vaneCounts = np.nanmean(vec1, axis=0)


    timestamps = integ1.data['DATE-OBS']
    mjd = np.mean(np.array([pipe.pu.dateToMjd(stamp)
                            for stamp in timestamps]))

    elevation = np.mean(integ1.data['ELEVATIO'])

    # Pull warm load temperature from the data
    twarm = np.mean(integ1.data['TWARM']+273.15)
    tbg = 2.725  # It's the CMB
    tambient = np.mean(integ1.data['TAMBIENT'])
    avgfreq = np.mean(integ1.data['OBSFREQ'])

    # Use weather model to get the atmospheric temperature and opacity
    if opacity:
        zenithtau = weather.retrieve_zenith_opacity(mjd, avgfreq, log=log,
                                                    forcecalc=True,
                                                    request='Opacity')
        tatm = weather.retrieve_Tatm(mjd, avgfreq, log=log,
                                     forcecalc=True)
        if not zenithtau:
            log.logger.warning('No weather data found in GBTWEATHER directory')
            log.logger.warning('Setting Zenith opacity to zero!')
            log.logger.warning('Temperatures on TA scale')
            zenithtau = 0
            tatm = tambient
    else:
        zenithtau = 0
        tatm = tambient
        log.logger.warning('Setting Zenith opacity to zero!')
        log.logger.warning('Temperatures on TA scale')
    # This does the airmass calculation
    tau = cal.elevation_adjusted_opacity(zenithtau, elevation)
    tcal = (tatm - tbg) + (twarm - tatm) * np.exp(tau)
    tsysStar = tcal / onoff
    return tcal, vaneCounts, tsysStar

def ZoneOfAvoidance(integrations, center=None,
                    radius=1 * u.arcmin, off_frac=0.1, **kwargs):
    """
    This function defines the OFFs based on an angular distance from a
    center point (i.e., a circular zone of avoidance)

    Keywords
    --------
    center : astropy.SkyCoord
        Centre position to avoid
    size : astropy.units
        Distance away from centre to be avoided in finding OFFs
    """
    frame_name = integrations.data['RADESYS'][0].astype(str).strip().lower()
    coords = SkyCoord(integrations.data['CRVAL2'],
                      integrations.data['CRVAL3'],
                      unit=(u.deg, u.deg),
                      frame=frame_name)
    offset = coords.separation(center)
    OffMask = offset > radius
    if np.all(~OffMask):
        warnings.warn("No scans found that are outside zone of avoidance")
        warnings.warn("Using row ends")
        OffMask, _ = RowEnds(integrations, off_frac=off_frac)
        return(OffMask, 'RowEnds')
    return(OffMask, 'ZoneOfAvoidance')

def FrequencySwitch(integrations):
    raise(NotImplementedError)

def SpatialMask(integrations, mask=None, wcs=None, off_frac=0.25, **kwargs):
    """
    This function defines the OFFs based on where they land in a
    spatial mask. For performance reasons, it's good import this

    Keywords
    --------
    mask : np.array
        mask image
    wcs : astropy.wcs.WCS
        WCS object corresponding to the mask

    """
    x, y = wcs.celestial.wcs_world2pix(integrations.data['CRVAL2'],
                                       integrations.data['CRVAL3'], 0)
    inx = np.logical_and((x >= 0), (x <= (mask.shape[1] - 1)))
    iny = np.logical_and((y >= 0), (y <= (mask.shape[0] - 1)))
    inarr = np.logical_and(inx, iny)
    OffMask = np.ones(len(x), dtype=bool)
    # Masks show where the emission IS and OffMask wants
    # the points where the mask IS NOT.  Hence the ~
    OffMask[inarr] = np.array(~mask[y[inarr].astype(int),
                                    x[inarr].astype(int)], dtype=bool)
    if np.all(~OffMask):
        warnings.warn("No scans found that are outside mask.")
        warnings.warn("Using row ends")
        OffMask, _ = RowEnds(integrations, off_frac=off_frac)
        return(OffMask, 'RowEnds')
    return(OffMask, 'SpatialMask')

def SpatialSpectralMask(integrations, mask=None, wcs=None,
                        off_frac=0.25, floatvalues=False, offpct=50,
                        **kwargs):
    scanshape = integrations.data['DATA'].shape # Nscans x Nchans
    OffMask = np.array(scanshape, dtype=bool)
    freq = ((np.linspace(1, scanshape[1], scanshape[1])[np.newaxis, :]
            - integrations.data['CRPIX1'][:, np.newaxis])
            * integrations.data['CDELT1'][:, np.newaxis]
            + integrations.data['CRVAL1'][:, np.newaxis])
    x, y, z = wcs.wcs_world2pix(integrations.data['CRVAL2'][:, np.newaxis],
                                integrations.data['CRVAL3'][:, np.newaxis],
                                freq, 0)
    y = np.clip(y, 0, mask.shape[1] - 1)
    x = np.clip(x, 0, mask.shape[2] - 1)
    z = np.clip(z, 0, mask.shape[0] - 1)
    badx = ~np.isfinite(x)
    bady = ~np.isfinite(y)
    badz = ~np.isfinite(z)
    x[badx] = 0
    y[bady] = 0
    z[badz] = 0
    # OffMask = True where OFF the galaxy)
    if floatvalues:
        OffEmission = np.array(mask[z.astype(int),
                                    y.astype(int),
                                    x.astype(int)],
                               dtype=float)
        OffMask = np.zeros_like(OffEmission, dtype=bool)
        EmScans = np.sum(OffEmission, axis=1)
        BetterScans = (EmScans <= np.percentile(EmScans, offpct))
        OffMask = (BetterScans[:, np.newaxis]
                   * np.ones((1, OffEmission.shape[1]), dtype=bool))
        # blank_chans = np.all(OffEmission == 0, axis=0)
        AllOn = np.all(~OffMask, axis=0)
    else:
        OffMask = np.array(mask[z.astype(int),
                                y.astype(int),
                                x.astype(int)], dtype=bool)
        OffMask = ~OffMask
        AllOn = np.all(~OffMask, axis=0)

        # mask[x.astype(int)[badx], y.astype(int)[bady]] = False
    if np.any(AllOn):
        warnings.warn("Some channels always on emission")
        OffMask[:, AllOn] = True
    
    return(OffMask, 'SpatialSpectralMask')



def NoMask(integrations, **kwargs):
    scanshape = integrations.data['DATA'].shape # Nscans x Nchans
    OffMask = np.ones(scanshape, dtype=bool)
    return(OffMask, 'NoMask')


def RowEnds(integrations, off_frac=0.25, **kwargs):
    """This function defines the OFFs as the ends of the rows
    
    Keywords
    --------
    off_frac : float
         Fraction of bandpass used to determine the off spectrum on either side.
    """
    nIntegrations = len(integrations.data)
    OffMask = np.zeros(nIntegrations, dtype=bool)
    OffMask[0:int(off_frac*nIntegrations)] = True
    OffMask[-int(off_frac*nIntegrations):] = True
    return(OffMask, 'RowEnds')

def calscans(inputdir, start=82, stop=105, refscans=[80],
             badscans=[], badfeeds=[],
             outdir=None, log=None, loglevel='warning',
             OffSelector=RowEnds, OffType='linefit',
             verbose=True, suffix='', nProc=1,
             opacity=True, varfrac=0.05, drop_last_scan=False,
             varrat=None, 
             smoothpca=False,
             **kwargs):
    """Main calibration routine

    Parameters
    ----------
    inputdir : str
        Name of input directory to chomp
    
    Keywords
    --------
    start : int
         Number of start scan
    stop : int
         Number of stop scan
    refscans : list of ints
         Reference scans to include in the calibration
    outdir : str
         Where to put the calibrated data
    log : gbtpipe.Logger
         Logging object
    loglevel : str
         Passed to log object to set verbosity
    OffType : str
         Select off model 'linefit' fits a line to the off
         setctions. 'median' uses the median of each part of the
         bandwpass
    verbose : bool
         Notify after every scan.
    suffix : str
         String to append onto every filename generated by this pipeline
    badscans : list of int
         List of scans to ignore in reduction
    badfeeds : list of int 
         List of feeds to ignore in reduction.  THIS IS ZERO INDEXED
         BECAUSE SOFTWARE.  We love hardware people but they count funny.
    drop_last_scan : bool
         Drop the last scan of each row. Needed for some modes.  
         Try with and without and see how it looks
    varfrac : float (optional)
         The fraction of variance to retain in dimensionality reduction. Default is 1e-4.
    varrat : float or None (optional)
         The ratio of variance between successive components to retain. 
         If None, no specific ratio is enforced. Default is None and PCA will use varfrac.
    smoothpca : bool
         Whether to apply time smoothing for PCA components (Principal Component Analysis). 
         Default is False.
    """
    
    # Grab them files
    if os.path.isdir(inputdir):
        fitsfiles = glob.glob(inputdir + '/*fits')
        if len(fitsfiles) == 0:
            warnings.warn("No FITS files found in input directory")
            return False
    elif os.path.isfile(inputdir):
        warnings.warn("Input name is a file and not a directory.")
        warnings.warn("Blindly reducing everything in the same directory.")
        infilename = inputdir
        inputdir = os.getcwd()
    else:
        warnings.warn("No file or directory found for inputs")
        return False

    if not outdir:
        outdir = os.getcwd()

    # Instantiate loggin
    makelogdir()
    if not log:
        log = Logging('gbtpipeline')
        if loglevel=='warning':
            log.logger.setLevel(30)

    # Grab the weather object
    w = Weather()
    
    # Start up the pipeline control object and feed in the propert scan numbers
    cl_params = initParameters(inputdir)
    # Assume spacing in uniform between start and stop
    cl_params.mapscans = list(np.linspace(start,stop,

                                          stop-start+1).astype('int'))
    for bad in badscans:
        if bad in cl_params.mapscans:
            cl_params.mapscans.remove(bad)
        
    cl_params.refscans = refscans
    for bad in badscans:
        if bad in cl_params.refscans:
            cl_params.refscans.remove(bad)
    # some logic should probably be added here if all ref scans are
    # bad, but in that case it would be a QA0 fail.

    if os.path.isdir(cl_params.infilename):
        log.doMessage('INFO', 'Infile name is a directory')
        input_directory = cl_params.infilename

        # Instantiate a SdFits object for I/O and interpreting the
        # contents of the index file
        sdf = SdFits()

        # generate a name for the index file based on the name of the
        # raw SDFITS file.  The index file simply has a different extension
        directory_name = os.path.basename(cl_params.infilename.rstrip('/'))
        indexfile = cl_params.infilename + '/' + directory_name + '.index'
        try:
            # create a structure that lists the raw SDFITS rows for
            #  each scan/window/feed/polarization
            row_list, summary = sdf.parseSdfitsIndex(indexfile,
                                                     cl_params.mapscans)
        except IOError:
            log.doMessage('ERR', 'Could not open index file', indexfile)
            log.close()
            return False
        # sys.exit()
    allfiles = glob.glob(input_directory + '/' +
                         os.path.basename(input_directory) +
                         '*.fits')
    for filectr, infilename in enumerate(allfiles):
            log.doMessage('DBG', 'Attempting to calibrate',
                          os.path.basename(infilename).rstrip('.fits'))

            # change the infilename in the params structure to the
            # current infile in the directory for each iteration
            cl_params.infilename = infilename

            # copy the cl_params structure so we can modify it during
            # calibration for each seperate file.
            command_options = copy.deepcopy(cl_params)
            sdf = SdFits()
            cal = Calibration()
            indexfile = sdf.nameIndexFile(command_options.infilename)
            row_list, summary = sdf.parseSdfitsIndex(indexfile,
                                                     mapscans=command_options.mapscans)
            feedlist = (row_list.feeds())

            # all feeds aren't going to be in the feedlist. The feedlist is per bank.
            for bad in badfeeds:
                if bad in feedlist:
                    feedlist.remove(bad)
            
            #Currently Argus only has one spectral window and polarization
            for thisfeed in feedlist:
                thispol = 0 # TODO: generalize to different POL/WIN
                thiswin = 0
                command_options = copy.deepcopy(cl_params)
                try:
                    pipe = MappingPipeline(command_options,
                                                   row_list,
                                                   thisfeed,
                                                   thispol,
                                                   thiswin,
                                                   None, outdir=outdir,
                                                   suffix=suffix,
                                                   log=log)
                except KeyError:
                    pipe = None
                    pass
                if pipe:                    
                    tcal, vaneCounts, tsysStar = gettsys(cl_params, row_list,
                                                         thisfeed, thispol,
                                                         thiswin, pipe,
                                                         weather=w, log=log,
                                                         opacity=opacity)
                    log.doMessage('INFO', 'Feed: {0}, Tsys (K): {1}'.format(
                            thisfeed,
                            tsysStar))


                    # pipe, row_list, thisscan, thisfeed, thispol, thiswin

                    onoffsets = []
                    for thisscan in cl_params.mapscans:
                        if verbose:
                            print("Now Processing Scan {0} for Feed {1}".format(
                                thisscan, thisfeed).ljust(50), end='\r')
                            sys.stdout.flush()
                        onoffsets.append(prepcal(thisscan, thisfeed=thisfeed,
                                                 thispol=thispol,
                                                 thiswin=thiswin, pipe=pipe,
                                                 row_list=row_list,
                                                 log=log, weather=w,
                                                 cal=cal,
                                                 OffSelector=OffSelector,
                                                 OffType=OffType,
                                                 tsysStar=tsysStar,
                                                 vaneCounts=vaneCounts,
                                                 cl_params=cl_params,
                                                 command_options=command_options,
                                                 allfiles=allfiles,
                                                 opacity=opacity,
                                                 varfrac=varfrac,
                                                 drop_last_scan=drop_last_scan,
                                                 tcal=tcal, **kwargs))
                    print('\n')
                        
                    calonoffsets = doOnOffAggregated(onoffsets, OffType=OffType, 
                                                     varfrac=varfrac, varrat=varrat, 
                                                     smoothpca=smoothpca)
                    # if nProc > 1:
                    #     with Pool(nProc) as pool:
                    #         calonoffsets = pool.map(doOnOff, onoffsets)
                    # else:
                    #     calonoffsets = []
                    #     for i, onoff in enumerate(onoffsets):
                    #         calonoffsets.append(doOnOff(onoff, OffType=OffType,
                    #                                     varfrac=varfrac))

                    for onoff in calonoffsets:
                        for ctr, rownum in enumerate(onoff['rows']):
                            # This updates the output SDFITS file with
                            # the newly calibrated data.
                            row = np.array([onoff['integs'].data[ctr]])
                            row['DATA'] = onoff['TAstar'][ctr,:]
                            row['TSYS'] = tsysStar
                            row['TUNIT7'] = 'Ta*'
                            pipe.outfile[-1].append(row)

                    pipe.infile.close()
                    pipe.outfile.close()
    return True

def prepcal(thisscan, thisfeed=0, thispol=0,
            thiswin=0, pipe=None, row_list=None, log=None,
            weather=None, cal=None, OffSelector=None, vaneCounts=None,
            tcal=None, tsysStar=None, cl_params=None, allfiles=None,
            command_options=None, OffType=None, opacity=True, drop_last_scan=False,
            **kwargs):
                        
    rows = row_list.get(thisscan, thisfeed,thispol, thiswin)
    ext = rows['EXTENSION']
    rows = rows['ROW']
    if drop_last_scan:
        rows.pop()
    columns = tuple(pipe.infile[ext].get_colnames())
    integs = ConvenientIntegration(
        pipe.infile[ext][columns][rows], log=log)
    # Grab everything we need to get a Tsys measure
    timestamps = integs.data['DATE-OBS']
    elevation = np.median(integs.data['ELEVATIO'])
    mjds = np.array([pipe.pu.dateToMjd(stamp)
                     for stamp in timestamps])
    avgfreq = np.median(integs.data['OBSFREQ'])

    # if opacity:
    #     zenithtau = weather.retrieve_zenith_opacity(np.median(mjds),
    #                                                 avgfreq,
    #                                                 log=log,
    #                                                 forcecalc=True)
    #     tau = cal.elevation_adjusted_opacity(zenithtau,
    #                                          elevation)
    # else:
    #     tau = 0
    #     log.log('Opacity set to zero.  All data on T_A scale only')
        
    # ARGUS beams 2 and 3 (software 1 and 2)
    # were swapped before 2018-10-22 19:30:00 UT

    if mjds[0] < 58413.81250000 and (thisfeed == 1):
        integs2 = findfeed(cl_params, allfiles,
                           command_options.mapscans,
                           thisscan, 2,
                           log=log)
        integs.data['CRVAL2'] = integs2.data['CRVAL2']
        integs.data['CRVAL3'] = integs2.data['CRVAL3']

    if mjds[0] < 58413.81250000 and (thisfeed == 2):
        integs2 = findfeed(cl_params, allfiles,
                           command_options.mapscans,
                           thisscan, 1,
                           log=log)
        integs.data['CRVAL2'] = integs2.data['CRVAL2']
        integs.data['CRVAL3'] = integs2.data['CRVAL3']

    # This block actually does the calibration
    ON = integs.data['DATA']
    # This identifies which scans to include as OFFs
    OffMask, OffStrategy = OffSelector(integs, **kwargs)
    if OffStrategy=='RowEnds' and OffType=='linefit':
        OffType = 'median'
    if OffStrategy=='SpatialMask' and OffType=='median':
        OffType = 'linefit'
    
    onoff = {'rows':rows,
             'integs':integs,
             'TAstar':ON,
             'tsysStar':tsysStar,
             'ON':ON,
             'OffMask':OffMask,
             'tcal':tcal,
             'vaneCounts':vaneCounts,
             'mjd':mjds}
    
    return(onoff)


def residual(params, data=None, components=None, mean=None):
    coeffs = []
    for key in params.keys():
        coeffs.append(params[key].value)
    coeffs = np.array(coeffs)
    pcavec = np.dot(coeffs, components) + mean
    return data - pcavec


def doOnOffAggregated(onoffset, OffType='PCA',
                      varfrac=1e-4, varrat=None, smoothpca=False):
    ONs = []
    splits = []
    ctr = 0
    for onoff in onoffset:
        ONs.append(onoff['ON'])
        ctr += onoff['ON'].shape[0]
        splits.append(ctr)
    
    ONs = [onoff['ON'] for onoff in onoffset]
    OffMasks = [onoff['OffMask'] for onoff in onoffset]
    vaneCounts = [onoff['vaneCounts'] for onoff in onoffset]
    mjds = [onoff['mjd'] for onoff in onoffset]

    ON = np.concatenate(ONs, axis=0)
    OffMask = np.concatenate(OffMasks, axis=0)
    mjds = np.concatenate(mjds, axis=0)

    ON = np.log(ON)
    if OffMask.ndim == 2:
        OffRows = np.all(OffMask, axis=1)
    else:
        OffRows = OffMask
    if OffType == 'PCA':
        # Use PCA to generate the components
        from sklearn.decomposition import PCA
        from lmfit import minimize, Parameters
        ncomp = 20
        ONselect = ON[np.where(OffMask)[0],:]
        mjds_select = mjds[OffRows]
        pcaobj = PCA(n_components=np.min([ncomp, ONselect.shape[0]-1]))
        pcaobj.fit(ONselect)
        
        coeffs = pcaobj.transform(ON)
        coeffs_select = pcaobj.transform(ONselect)
        MeanON = np.nanmean(ONselect, axis=0)
        if varrat:
            retain = np.sum(pcaobj.explained_variance_ratio_[0:-1] 
                            / pcaobj.explained_variance_ratio_[1:] > varrat)
        else:
            retain = np.sum(pcaobj.explained_variance_ratio_ > varfrac)
            
        if smoothpca:
            coeffs_smooth = np.zeros_like(coeffs)
            meantime = np.mean(mjds)
            for component in range(retain):
                smoothing_func = make_smoothing_spline(mjds_select - meantime, 
                                                       coeffs_select[:,component], 
                                                       lam=1e-12)
                coeffs_smooth[:, component] = smoothing_func(mjds - meantime)
            coeffs = coeffs_smooth
            
        AllOFF = (np.dot(coeffs[:, 0:retain],
                  pcaobj.components_[0:retain, :])
                  + MeanON)
        AllOFF = np.exp(AllOFF)
        ON = np.exp(ON)
        OFFs = np.split(AllOFF, splits, axis=0)
        
    for onoff, OFF in zip(onoffset, OFFs):
        ON = onoff['ON']
        onoff['OFF'] = OFF
        onoff['TAstar'] = (onoff['tcal'] * (ON - OFF) / OFF)
        medianOFF = np.nanmedian(OFF, axis=0)

        # Now construct a scalar factor by taking
        # median OFF power (over time) and compare to
        # the mean vaneCounts over time.  Then take
        # the median of this ratio and apply.

        scalarOFFfactor = np.median(medianOFF /
                                    (vaneCounts - medianOFF))
        TA = (onoff['tcal'] * scalarOFFfactor
            * (ON - OFF) / (OFF))
        medianTA = np.median(TA, axis=1)
        medianTA.shape = (1,) + medianTA.shape
        medianTA = np.ones((ON.shape[1], 1)) * medianTA
        TAstar = TA - medianTA.T
        onoff['TAstar'] = TAstar
    return(onoffset)


def doOnOff(onoff, OffType='PCA', OffStrategy='RowEnds', varfrac=0.05):
    ON = onoff['ON']
    OffMask = onoff['OffMask']
    vaneCounts = onoff['vaneCounts']

    if OffType == 'median2d':

        # This builds a 2D median map of a data set then does a
        # linear fit to the residual in the time axis to correct the residual.
        
        medianpow = np.nanmedian(ON, axis=1)
        medianpow /= np.nanmean(medianpow)
        medianON = np.nanmedian(ON, axis=0)
        OFF = (medianON[np.newaxis, :]
               * medianpow[:, np.newaxis])

        diff = (ON-OFF)
        xaxis = np.linspace(-0.5,0.5,diff.shape[0])
        xaxis.shape += (1,)
        xaxis = xaxis * np.ones((1,diff.shape[1]))
        xsub = xaxis
        ONsub = copy.deepcopy(diff)
        ONsub[~OffMask] = np.nan
        # This is a little tedious because we want
        # to vectorize the least-squares fit per
        # scan.
        # MeanON = np.nanmean(ONsub, axis=0)
        MeanON = np.nanmedian(ONsub, axis=0)
        MeanON.shape += (1,)
        MeanON = MeanON * np.ones((1, ONsub.shape[0]))
        MeanON = MeanON.T

        MeanX = np.nanmean(xsub, axis=0)
        MeanX.shape += (1,)
        MeanX = MeanX * np.ones((1, ONsub.shape[0]))
        MeanX = MeanX.T

        # The solution to all linear least-squares.
        slope = (np.nansum(((xsub - MeanX)
                            * (ONsub - MeanON)),
                           axis=0)
                 / np.nansum((xsub - MeanX)**2,
                             axis=0))
        slope.shape += (1,)
        slope = slope * np.ones((1, ON.shape[0]))
        MeanON = MeanON[0,:]
        MeanON.shape += (1,)
        MeanON = MeanON * np.ones((1,ON.shape[0]))
        MeanON = MeanON.T
        # This the off model.
        DiffCorr = slope.T * xaxis + MeanON
        OFF = OFF + DiffCorr

    if OffType == 'PCA':
        # Use PCA to generate the components
        from sklearn.decomposition import PCA

        ncomp = 10
        if OffMask.ndim == 1:
            ONselect = ON[np.where(OffMask)[0],:]
        else:
            ONselect = ON[np.where(OffMask[:,0])[0],:]
        pcaobj = PCA(n_components=np.min([ncomp, ONselect.shape[0]-1]),
                     svd_solver='full')
        pcaobj.fit(ONselect)
        # pcaobj.fit(np.r_[ONselect,
        #                  np.roll(ONselect, 1, axis=1),
        #                  np.roll(ONselect, -1, axis=1)])
        
        coeffs = pcaobj.transform(ON)
        MeanON = np.nanmean(ONselect,axis=0)
        retain = np.sum(pcaobj.explained_variance_ratio_ > varfrac)
        OFF = (np.dot(coeffs[:, 0:retain],
                      pcaobj.components_[0:retain, :])
               + MeanON)

    if OffType == 'median':
        # Model the off power as the median counts
        # across the whole bandpass. This assumes
        # that line power is weak per scan and per
        # channel.
        medianOFF = np.nanmedian(ON, axis=0)
        OFF = np.median(ON[OffMask, :], axis=0) 
        # Empirical bandpass
        OFF.shape += (1,)
        OFF = OFF * np.ones((1, ON.shape[0]))
        OFF = OFF.T

    if OffType == 'linefit':
        # This block fits a line to the off scans 
        # in the bandpass as a model for the power.
        xaxis = np.linspace(-0.5,0.5,ON.shape[0])
        xaxis.shape += (1,)
        xaxis = xaxis * np.ones((1,ON.shape[1]))
        # xsub = xaxis[OffMask,:]
        # ONsub = ON[OffMask, :]
        xsub = xaxis
        ONsub = copy.deepcopy(ON)
        ONsub[~OffMask] = np.nan
        # This is a little tedious because we want
        # to vectorize the least-squares fit per
        # scan.
        MeanON = np.nanmean(ONsub, axis=0)
        # MeanON = np.nanmedian(ON, axis=0)
        MeanON.shape += (1,)
        MeanON = MeanON * np.ones((1, ONsub.shape[0]))
        MeanON = MeanON.T

        MeanX = np.nanmean(xsub, axis=0)
        MeanX.shape += (1,)
        MeanX = MeanX * np.ones((1, ONsub.shape[0]))
        MeanX = MeanX.T

        # The solution to all linear least-squares.
        slope = (np.nansum(((xsub - MeanX)
                            * (ONsub - MeanON)),
                           axis=0)
                 / np.nansum((xsub - MeanX)**2,
                             axis=0))
        slope.shape += (1,)
        slope = slope * np.ones((1, ON.shape[0]))
        MeanON = MeanON[0,:]
        MeanON.shape += (1,)
        MeanON = MeanON * np.ones((1,ON.shape[0]))
        MeanON = MeanON.T
        # This the off model.
        OFF = slope.T * xaxis + MeanON

    medianOFF = np.nanmedian(OFF, axis=0)

    # Now construct a scalar factor by taking
    # median OFF power (over time) and compare to
    # the mean vaneCounts over time.  Then take
    # the median of this ratio and apply.

    scalarOFFfactor = np.median(medianOFF /
                                (vaneCounts - medianOFF))
    TA = (onoff['tcal'] * scalarOFFfactor
          * (ON - OFF) / (OFF))
    medianTA = np.median(TA, axis=1)
    medianTA.shape = (1,) + medianTA.shape
    medianTA = np.ones((ON.shape[1], 1)) * medianTA
    TAstar = TA - medianTA.T
    onoff['TAstar'] = TAstar
    return(onoff)

