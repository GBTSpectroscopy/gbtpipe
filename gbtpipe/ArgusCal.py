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
import fitsio
import copy
import warnings
import sys
import astropy.units as u
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord

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
        cal = Calibration()
        indexfile = sdf.nameIndexFile(anotherfile)
        row_list, summary = sdf.parseSdfitsIndex(indexfile,
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
                                   suffix='_tmp')
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

    if ((('Vane' in str(integ1.data['CALPOSITION'][0],
                        encoding='UTF-8')) and
         ('Observing' in str(integ2.data['CALPOSITION'][0],
                             encoding='UTF-8')) or 
         (('VANE' in str(integ1.data['OBJECT'][0],
                         encoding='UTF-8')) and 
          ('SKY' in str(integ2.data['OBJECT'][0],
                        encoding='UTF-8'))))):
        # Directly do an on-off on the full data set
        onoff = np.median((vec1-vec2)/vec2)
        # Mean over time to get a vector of vaneCounts at each freq.
        vaneCounts = np.mean(vec1, axis=0)

    # Now test case where Vane data were second position
    elif ((('Vane' in str(integ2.data['CALPOSITION'][0],
                          encoding='UTF-8')) and
           ('Observing' in str(integ1.data['CALPOSITION'][0])) or 
          (('VANE' in str(integ2.data['OBJECT'][0],
                          encoding='UTF-8')) and 
           ('SKY' in str(integ1.data['OBJECT'][0],
                         encoding='UTF-8'))))):
        onoff = np.median((vec2-vec1)/vec1)
        vaneCounts = np.mean(vec2, axis=0)
    else:
        # If you are here, then you will not get data today
        warnings.warn("No Vane scans found in putative reference scans.")
        warnings.warn("Making a wild guess that's probably wrong...")
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
    zenithtau = weather.retrieve_zenith_opacity(mjd, avgfreq, log=log,
                                                forcecalc=True,
                                                request='Opacity')
    tatm = weather.retrieve_Tatm(mjd, avgfreq, log=log,
                                 forcecalc=True)

    # This does the airmass calculation
    tau = cal.elevation_adjusted_opacity(zenithtau, elevation)
    tcal = (tatm - tbg) + (twarm - tatm) * np.exp(tau)
    tsysStar = tcal / onoff
    return tcal, vaneCounts, tsysStar

def ZoneOfAvoidance(integrations, center=None,
                    radius=1 * u.arcmin, off_frac=0.1):
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
    frame_name = integrations.data['RADESYS'][0].astype(np.str).strip().lower()
    coords = SkyCoord(integrations.data['CRVAL2'],
                      integrations.data['CRVAL3'],
                      unit=(u.deg, u.deg),
                      frame=frame_name)
    offset = coords.separation(center)
    OffMask = offset > radius
    if np.all(~OffMask):
        warnings.warn("No scans found that are outside zone of avoidance")
        warnings.warn("Using row ends")
        OffMask = RowEnds(integrations, off_frac=off_frac,
                          exclude_frac=off_frac/2)
        return(OffMask)
    return(OffMask)

def FrequencySwitch(integrations):
    raise(NotImplementedError)

def SpatialMask(integrations, mask=None, wcs=None):
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
    OffMask = np.array(mask[y.astype(np.int), x.astype(np.int)], dtype=np.bool)
    if np.all(~OffMask):
        warnings.warn("No scans found that are outside zone of avoidance")
        warnings.warn("Using row ends")
        OffMask = RowEnds(integrations, off_frac=off_frac,
                          exclude_frac=off_frac/2)
        return(OffMask)
    return(OffMask)

def RowEnds(integrations, off_frac=0.25):
    """This function defines the OFFs as the ends of the rows
    
    Keywords
    --------
    off_frac : float
         Fraction of bandpass used to determine the off spectrum on either side.
    """
    nIntegrations = len(integrations.data)
    OffMask = np.zeros(nIntegrations, dtype=np.bool)
    OffMask[0:int(off_frac*nIntegrations)] = True
    OffMask[-int(off_frac*nIntegrations):] = True
    return(OffMask)

def calscans(inputdir, start=82, stop=105, refscans=[80],
             outdir=None, log=None, loglevel='warning',
             OffSelector=RowEnds, OffType='linefit',
             verbose=True, suffix='', **kwargs):
    """
    Main calibration routine

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
    cl_params.refscans = refscans
    
    
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
                                                   suffix=suffix)
                except KeyError:
                    pipe = None
                    pass
                if pipe:
                    tcal, vaneCounts, tsysStar = gettsys(cl_params, row_list,
                                                         thisfeed, thispol,
                                                         thiswin, pipe,
                                                         weather=w, log=log)
                    log.doMessage('INFO', 'Feed: {0}, Tsys (K): {1}'.format(
                            thisfeed,
                            tsysStar))

                    for thisscan in cl_params.mapscans:
                        if verbose:
                            print("Now Processing Scan {0} for Feed {1}".format(
                                thisscan, thisfeed).ljust(50), end='\r')
                            sys.stdout.flush()

                        rows = row_list.get(thisscan, thisfeed,
                                            thispol, thiswin)
                        ext = rows['EXTENSION']
                        rows = rows['ROW']
                        columns = tuple(pipe.infile[ext].get_colnames())
                        integs = ConvenientIntegration(
                            pipe.infile[ext][columns][rows], log=log)
                        
                        # Grab everything we need to get a Tsys measure
                        timestamps = integs.data['DATE-OBS']
                        elevation = np.median(integs.data['ELEVATIO'])
                        mjds = np.array([pipe.pu.dateToMjd(stamp)
                                         for stamp in timestamps])
                        avgfreq = np.median(integs.data['OBSFREQ'])
                        zenithtau = w.retrieve_zenith_opacity(np.median(mjds),
                                                              avgfreq,
                                                              log=log,
                                                              forcecalc=True)
                        tau = cal.elevation_adjusted_opacity(zenithtau,
                                                             elevation)

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
                        OffMask = OffSelector(integs, **kwargs)
                        if OffType == 'median':
                            # Model the off power as the median counts
                            # across the whole bandpass. This assumes
                            # that line power is weak per scan and per
                            # channel.
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
                            xsub = xaxis[OffMask,:]
                            ONsub = ON[OffMask, :]

                            # This is a little tedious because we want
                            # to vectorize the least-squares fit per
                            # scan.
                            MeanON = np.nanmean(ONsub, axis=0)
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
                        TA = (tcal * scalarOFFfactor
                              * (ON - OFF) / (OFF))
                        medianTA = np.median(TA, axis=1)
                        medianTA.shape = (1,) + medianTA.shape
                        medianTA = np.ones((ON.shape[1], 1)) * medianTA
                        TAstar = TA - medianTA.T
                        for ctr, rownum in enumerate(rows):
                            # This updates the output SDFITS file with
                            # the newly calibrated data.

                            # row = Integration(
                            #     pipe.infile[ext][columns][rownum])
                            row = np.array([integs.data[ctr]])
                            row['DATA'] = TAstar[ctr,:]
                            row['TSYS'] = tsysStar
                            row['TUNIT7'] = 'Ta*'
                            pipe.outfile[-1].append(row)
                    pipe.infile.close()
                    pipe.outfile.close()
    return True
