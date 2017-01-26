
from Pipeutils import Pipeutils
import numpy as np
from Integration import Integration
import Weather
from PipeLogging import Logging
class ConvenientIntegration(Integration):
    def __init__(self, row, log=None):
        self.pu = Pipeutils()
        self.data = row
        self.weather = Weather.Weather()
        if not log:
            self.log = Logging()
    def mjd(self):
        return self.pu.dateToMjd(self.data['DATE-OBS'][0])
    def obsFreq(self):
        return self.data['OBSFREQ'][0]
    def zenith_tau(self):
        return self.weather.retrieve_zenith_opacity(self.mjd(), 
                                                    self.obsFreq(),
                                                    log=self.log,
                                                    forcecalc=True)


