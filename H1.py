# Code to aid analysis of H1 data - Esha Sajjanhar
# Spring 2024, Ashoka University

import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
import astropy.units as u
from astropy.time import Time
pi = np.pi

def load_file(file_name, start_freq=1419405751, stop_freq=1421405751, freq_step=3906.25):
    ''' Load a file recorded using the IUCAA GUI and return the frequency, power level for the recorded
    signal.
    
    Parameters:
    • file_name : str
    The file path leading to the '.csv' file containing the recorded signal.
    • start_freq : int, default: 1419405751
    The starting frequency for the band being considered.
    • stop_freq : int, default: 1421405751
    The last frequency for the band being considered.
    • freq_step : float, default: 3906.25
    The frequency step using which the data has been collected.

    Returns:
    • freq : array_like
    The frequencies in the recorded band [start_freq, stop_freq] each separated by freq_step.
    • power : array_like
    The recorded power levels for each freqency in the recorded band.

    Usage:
    >>> freq, power_source = load_file('./source.csv')
    '''
    
    n_cols = int((stop_freq-start_freq)/freq_step)
    freq = np.arange(start_freq, stop_freq+freq_step, freq_step)
    cols = np.arange(6, n_cols+7, 1)
    power = np.loadtxt(file_name, delimiter=',', usecols=cols, unpack=True)
    return freq, power

def to_equatorial(az, alt, time=False, loc=False, output_obj=False, degree=True):
    ''' Convert given local coordinates to equatorial coordinates given the time and location of observation.
    Parameters:
    • az : int
    The azimuth (in degrees) to be converted.
    • alt : int
    The altitude (in degrees) coordinate to be converted.
    • time : tuple, default: False
    By default, the time of observation is taken to be the current time. To enter a different time, a tuple of the format (year, month, date, hours, minutes, seconds) can be used.
    • loc : tuple, default: False
    By default, the location of observation is taken to be Sonipat. To enter a different time, a tuple of the format (latitude, longitude, height) can be used.
    • obj_output : bool, default: False
    A boolean variable to choose whether the output is an AstroPy coordinate class or float. By default, the output is two float coordinates.
    • degree : bool, default: True
    A boolean variable to choose whether the output should be in radians or degrees. By default, the output is in degrees.
    Returns:
    When output_obj=False:
    • RA: float
    The right ascension of the object at the entered local coordinates.
    • dec: float
    The declination of the object at the entered local coordinates.
    When output_obj=True:
    • equatorial: astropy.coordinates.sky_coordinate.SkyCoord
    An AstroPy coordinate class containing the equatorial coordinates of the object at the entered local coordinates.
    
    Usage:
    ra_deg, dec_deg = to_equatorial(300, 53, time=(2024, 1, 23, 13, 40, 0))
    '''
    utcoffset = +5.5*u.hour #Eastern Daylight Time

    if loc:
        lat, lon, h = loc
        location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=h*u.m)
    else:
        location = EarthLocation(lat=28.9931*u.deg, lon=77.0151*u.deg, height=224*u.m)

    if time:
        time_utc = Time(datetime.datetime(*time)) - utcoffset
        horizontal = SkyCoord(AltAz(alt=alt*u.degree, az=az*u.degree, location=location, obstime=time_utc))
        equatorial = horizontal.transform_to('icrs')
    else:
        current_time = datetime.datetime.now()
        time_utc = Time(current_time) - utcoffset

        horizontal = SkyCoord(AltAz(alt=alt*u.degree, az=az*u.degree, location=location, obstime=time_utc))
        equatorial = horizontal.transform_to('icrs')

    if output_obj:
        return equatorial
    else:
        if degree:
            return equatorial.ra.degree, equatorial.dec.degree
        else:
            return equatorial.ra.radian, equatorial.dec.radian

def to_galactic(az, alt, time=False, loc=False, output_obj=False, degree=True):
    ''' Convert given local coordinates to galactic coordinates given the time and location of observation.
    Parameters:
    • az : int
    The azimuth (in degrees) to be converted.
    • alt : int
    The altitude (in degrees) to be converted.
    • time : tuple, default: False
    By default, the time of observation is taken to be the current time. To enter a different time, a tuple of the format (year, month, date, hours, minutes, seconds) can be used.
    • loc : tuple, default: False
    By default, the location of observation is taken to be Sonipat. To enter a different time, a tuple of the format (latitude, longitude, height) can be used.
    • obj_output : bool, default: False
    A boolean variable to choose whether the output is an AstroPy coordinate class or float. By default, the output is two float coordinates.
    • degree : bool, default: True
    A boolean variable to choose whether the output should be in radians or degrees. By default, the output is in degrees.
    Returns:
    When output_obj=False:
    • l: float
    The galactic longitude of the object at the entered local coordinates.
    • b: float
    The galactic latitude of the object at the entered local coordinates.
    When output_obj=True:
    • galactic: astropy.coordinates.sky_coordinate.SkyCoord
    An AstroPy coordinate class containing the equatorial coordinates of the object at the entered local coordinates.
    
    Usage:
    l_deg, b_deg = to_galactic(300, 53, time=(2024, 1, 23, 13, 40, 0))
    '''
    utcoffset = +5.5*u.hour #Eastern Daylight Time

    if loc:
        lat, lon, h = loc
        location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=h*u.m)
    else:
        location = EarthLocation(lat=28.9931*u.deg, lon=77.0151*u.deg, height=224*u.m)

    if time:
        time_utc = Time(datetime.datetime(*time)) - utcoffset
        horizontal = SkyCoord(AltAz(alt=alt*u.degree, az=az*u.degree, location=location, obstime=time_utc))
        galactic = horizontal.transform_to('galactic')
    else:
        current_time = datetime.datetime.now()
        time_utc = Time(current_time) - utcoffset

        horizontal = SkyCoord(AltAz(alt=alt*u.degree, az=az*u.degree, location=location, obstime=time_utc))
        galactic = horizontal.transform_to('galactic')
    if output_obj:
        return galactic
    else:
        if degree:
            return galactic.l.degree, galactic.b.degree
        else:
            return galactic.l.radian, galactic.b.radian
    
    
def lsr(az, alt, time, location=EarthLocation(lat=28.9931*u.deg, lon=77.0151*u.deg, height=224*u.m)):
    '''Compute the local standard of rest (LSR) correction for a given observation.
    Parameters:
    • az : int
    The azimuth (in degrees) for the observation.
    • alt : int
    The altitude (in degrees) for the observation.
    • time : tuple
    The time of observation in the format (year, month, date, hours, minutes, seconds) can be used.
    • loc : tuple, default: False
    By default, the location of observation is taken to be Sonipat. To enter a different time, a tuple of the format (latitude, longitude, height) can be used.
    
    Returns:
    • lsr: float
    The lsr correction to be added to the velocity of the source observed.
    
    Usage:
    lsr_corr = lsr(300, 53, (2024, 1, 23, 13, 40, 0))
    '''
    equatorial = to_equatorial(az, alt, time=time, output_obj=True, degree=False)
    ra_rad = equatorial.ra.radian
    dec_rad = equatorial.dec.radian
    
    if time==False:
        barycorr = equatorial.radial_velocity_correction(obstime=Time(datetime.datetime.now()))#, location=sonipat)
        barycorr = barycorr.to(u.km/u.s)
        heliocorr = equatorial.radial_velocity_correction('heliocentric', obstime=Time(datetime.datetime.now()))#, location=sonipat)
        heliocorr = heliocorr.to(u.km/u.s)
    else:
        barycorr = equatorial.radial_velocity_correction(obstime=Time(datetime.datetime(*time)))#, location=sonipat)
        barycorr = barycorr.to(u.km/u.s)
        heliocorr = equatorial.radial_velocity_correction('heliocentric', obstime=Time(datetime.datetime(*time)))#, location=sonipat)
        heliocorr = heliocorr.to(u.km/u.s)

    v_sun = 20.5*(u.km/u.s)
    sun_ra = (270*pi/180)*u.radian
    sun_dec = (30*pi/180)*u.radian

    a = np.cos(sun_dec)*np.cos(dec_rad)
    b = (np.cos(sun_ra)*np.cos(ra_rad)) + (np.sin(sun_ra)*np.sin(ra_rad))
    c = np.sin(sun_dec)*np.sin(dec_rad)
    v_rs = v_sun*(a*b + c)
    lsr =  v_rs + heliocorr
    
    return lsr
    
def baseline_sub(vel, temp, index1, index2):
    '''Compute and subtract a polynomial baseline from the velocity profile.
    Parameters:
    • vel : array_like
    The array of velocities (after lsr correction) in the velocity-temperature profile for a given source.
    • temp : array_like
    The array of temperatures in the velocity-temperature profile for a given source.
    • index1 : int
    The array index at which to clip the temperature array before baseline subtraction. This should be pick such that the clipped array (temp[index1:index2]) does not contain any part of the H1 signal peak.
    • index2 : int
    The array index at which to clip the temperature array before baseline subtraction. This should be pick such that the clipped array (temp[index1:index2]) does not contain any part of the H1 signal peak.
    
    Returns:
    • temp_sub: array_like
    The temperature array for the velocit-temperature profile after polynomial baseline subtraction.
    
    Usage:
    T_final = baseline_sub(vel, temp, 180, 350)
    '''
    def cubic(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d
    
    clip = np.concatenate((vel[:index1], vel[index2:]))
    T = np.concatenate((temp[:index1], temp[index2:]))
    args = np.polyfit(clip, T, 3)
    temp_sub = temp - np.array(cubic(vel, *args))
    
    return temp_sub