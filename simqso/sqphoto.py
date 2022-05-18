#!/usr/bin/env python

import os
import numpy as np
from collections import OrderedDict
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.constants import c
c_Angs = c*1e10

from astropy.io import fits
from astropy.table import Table

from .sqbase import datadir

softening_parameter = np.array([1.4,0.9,1.2,1.8,7.4])*1e-10

def nmgy2abmag(_b,f,df=None):
    ii = np.where(f>0)
    mag = 99.99 + np.zeros_like(f)
    mag[ii] = 22.5 - 2.5*np.log10(f[ii])
    if df is None:
        return mag
    else:
        err = np.zeros_like(mag)
        err[ii] = 1.0857 * df[ii]/f[ii]
        return mag,err

def abmag2nmgy(_b,m):
    return 10**(-0.4*(m - 22.5))

def nmgy2asinhmag(_b,f,df=None):
    b = softening_parameter['ugriz'.find(_b)]
    mag = -1.0857*(np.arcsinh(1e-9*f/(2*b)) + np.log(b))
    if df is None:
        return mag
    else:
        err = 1.0857 * 1e-9*df/(2*b) / np.sqrt(1+((1e-9*f)/(2*b))**2)
        return mag,err

def asinhmag2nmgy(_b,m):
    b = softening_parameter['ugriz'.find(_b)]
    return 2*b*np.sinh(m/(-1.0857) - np.log(b)) / 1e-9

# Vega...

#
# SDSS photometry model
#

_sdss_phot_pars = {
  'gain':[     1.6,   3.925,  4.7225,    4.86,    4.76, ],
  'darkVariance':[ 9.45625, 1.63125, 1.16125,    6.25,   1.105, ],
  'sky':[ 1.33136, 1.70364, 4.35521, 8.10383, 25.3321, ],
  'skyErr':[ 0.00657128, 0.00254991, 0.00409365, 0.00670879, 0.0255355, ],
  'nEffPsf':[ 36.4706, 32.1649, 27.2765,  24.854, 25.9643, ],
  'nMgyPerCount':[ 0.00981, 0.00378, 0.00507, 0.00662, 0.0337, ],
}

class sdssPhotoUnc(object):
    '''
    In a given SDSS band "b", provide the uncertainty for a given flux in
    nanomaggies (f_nmgy) based on the distribution of observing conditions.

    --> Currently underestimates true scatter by using gaussians for the
    scatter, whereas the true distributions generally have long tails
    to higher values for sky, nEff, etc.

    see http://classic.sdss.org/dr7/algorithms/fluxcal.html for details
    '''
    def __init__(self,b):
        i = 'ugriz'.find(b)
        self.pixArea = 0.396**2 # pix -> arcsec^2
        self.gain = _sdss_phot_pars['gain'][i]
        self.darkVar = _sdss_phot_pars['darkVariance'][i]
        self.skyMean = _sdss_phot_pars['sky'][i]
        self.skyStd = {'u':0.4,'g':0.4,'r':1.2,'i':2.0,'z':5.0}[b]
        self.skyMin = {'u':0.6,'g':1.0,'r':2.2,'i':3.2,'z':8.3}[b]
        #skyErr_nmgy = _sdss_phot_pars['skyErr'][i] # not used...
        # nEffPsf distribution is roughly similar in all bands
        self.npixMean = _sdss_phot_pars['nEffPsf'][i]
        self.npixStd = 5.0
        self.npixMin = 10.0
        self.c2fMean = _sdss_phot_pars['nMgyPerCount'][i]
        self.c2fStd = {'u':2.3e-3,'g':3.9e-4,'r':2.7e-4,
                       'i':3.7e-4,'z':5.6e-3}[b]
        self.c2fMin = {'u':5.7e-3,'g':2.3e-3,'r':3.5e-3,
                       'i':4.7e-3,'z':1.4e-2}[b]
        # add in a global photometric calibration error term
        self.calibrationError = 0.015
    def __call__(self,f_nmgy):
        shape = f_nmgy.shape
        gain = self.gain
        pixArea = self.pixArea
        darkVar = self.darkVar
        sky_nmgy_asec2 = np.clip(np.random.normal(self.skyMean,
                                                  self.skyStd,shape),
                                 self.skyMin,np.inf)
        npix = np.clip(np.random.normal(self.npixMean,self.npixStd,shape),
                       self.npixMin,np.inf)
        c2f = np.clip(np.random.normal(self.c2fMean,self.c2fStd,shape),
                      self.c2fMin,np.inf)
        df = np.sqrt( f_nmgy*(c2f/gain) +
                       sky_nmgy_asec2*pixArea*npix*(c2f/gain) +
                        darkVar*npix*(c2f/gain)**2 +
                         (self.calibrationError*f_nmgy)**2 )
        return df

class empiricalPhotoUnc(object):
    '''approximation only valid in sky-dominated regime'''
    def __call__(self,f_nmgy):
        shape = f_nmgy.shape
        # set the flux for non-detections to be at d(mag) = 1.0
        magLim = self.b / self.a
        magAB = np.clip(nmgy2abmag(self.b,f_nmgy),0,magLim)
        scatter = np.clip(self.scatter_a*magAB + self.scatter_b, 0.01, np.inf)
        b = self.b + scatter*np.random.normal(size=shape)
        log_dm = 2.5*(self.a*magAB - b)
        dm = np.clip(10**log_dm, self.err_floor, np.inf)
        f = np.clip(f_nmgy,abmag2nmgy(self.band,magLim),np.inf)
        return f * dm / 1.0857

class ukidsslasPhotoUnc(empiricalPhotoUnc):
    def __init__(self,b):
        UKIDSS_LAS_terms = np.array([[0.13616,3.1360,0.029],
                                     [0.14665,3.3081,0.043],
                                     [0.14429,3.2105,0.040],
                                     [0.15013,3.3053,0.028]])
        self.band = b
        i = 'YJHK'.find(b)
        self.a,self.b,self.scatter_b = UKIDSS_LAS_terms[i]
        # ignoring magnitude-dependent scatter since all useful fluxes are
        # in the sky-dominated regime
        self.scatter_a = 0.0
        # scatter seems to be slightly overestimated
        self.scatter_b *= 0.9
        # calibration uncertainty floor
        self.err_floor = 0.015

class ukidssdxsPhotoUnc(empiricalPhotoUnc):
    '''as with Stripe82, not valid at bright magnitudes (m<~20)'''
    def __init__(self,b):
        UKIDSS_DXS_terms = np.array([[0.13408,3.3978,0.016],
                                     [0.14336,3.5461,0.023]])
        self.band = b
        i = 'JK'.find(b)
        self.a,self.b,self.scatter_b = UKIDSS_DXS_terms[i]
        # ignoring magnitude-dependent scatter since all useful fluxes are
        # in the sky-dominated regime
        self.scatter_a = 0.0
        # scatter seems to be slightly overestimated again (?)
        self.scatter_b *= 0.8
        # calibration uncertainty floor
        self.err_floor = 0.015

class sdssStripe82PhotoUnc(empiricalPhotoUnc):
    '''this fails at m<~18 when SDSS detections are no longer sky-dominated,
       but not really interested in bright objects on the Stripe...
       also, dominated by calibration uncertainty for bright objects anyway
    '''
    def __init__(self,b):
        stripe82terms = np.array([[0.15127,3.8529,0.00727,-0.1308],
                                  [0.15180,4.0233,0.00486,-0.0737],
                                  [0.14878,3.8970,0.00664,-0.1077],
                                  [0.14780,3.8024,0.00545,-0.0678],
                                  [0.14497,3.5437,0.00715,-0.1121]])
        self.band = b
        i = 'ugriz'.find(b)
        self.a,self.b,self.scatter_a,self.scatter_b = stripe82terms[i]
        # calibration uncertainty floor
        self.err_floor = 0.015

class cfhtlsWidePhotoUnc(empiricalPhotoUnc):
    '''as with Stripe82, not valid at bright magnitudes (m<~19)'''
    def __init__(self,b):
        cfhtlswideterms = np.array([[0.16191,4.4005,0.037],
                                    [0.15508,4.3392,0.034],
                                    [0.15902,4.3399,0.015],
                                    [0.15721,4.2786,0.028],
                                    [0.16092,4.1967,0.034]])
        self.band = b
        i = 'ugriz'.find(b)
        self.a,self.b,self.scatter_b = cfhtlswideterms[i]
        # ignoring magnitude-dependent scatter since all useful fluxes are
        # in the sky-dominated regime
        self.scatter_a = 0.0
        # calibration uncertainty floor
        self.err_floor = 0.015


# WISE photometric model
# need to find original reference, values updated to AllWISE by Feige

_wise_phot_pars = {
  'n':{'W1':1.10688849e-08, 'W2':4.80175292e-08,
       'W3':4.4392921685e-06, 'W4':0.000104547518424 },
  'n_lo':{'W1':8.54987472e-09, 'W2':3.69296068e-08, },
  'n_hi':{'W1':1.35878952e-08, 'W2':5.91054515e-08, },
  'a':{'W1':2.22977172e-02, 'W2':1.92250161e-02,
       'W3':0.01, 'W4':0.01},
  'a_lo':{'W1':2.18542185e-02, 'W2':1.89920736e-02, },
  'a_hi':{'W1':2.27412159e-02, 'W2':1.94579587e-02, },
  # http://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec4_3g.html#WISEZMA
  'ABtoVega':{'W1':2.699,'W2':3.339,'W3':5.174,'W4':6.620},
}

class allwisePhotoUnc(object):
    def __init__(self,b):
        self.band = b
        self.n = _wise_phot_pars['n'][b]
        self.n_lo = _wise_phot_pars['n_lo'][b]
        self.n_hi = _wise_phot_pars['n_hi'][b]
        self.a = _wise_phot_pars['a'][b]
        self.a_lo = _wise_phot_pars['a_lo'][b]
        self.a_hi = _wise_phot_pars['a_hi'][b]
        self.vegaConv = _wise_phot_pars['ABtoVega'][b]
    def __call__(self,f_nmgy):
        vegaMag = nmgy2abmag(self.band,f_nmgy) - self.vegaConv
        sig_m = self.a + 1.0857*self.n/(10**(-0.4*vegaMag))
        s = f_nmgy.shape
        try:
            # this is legacy code for approximating the depth variations
            # over the sky, not really sure how valid it is
            sig_m_lo = self.a_lo + 1.0857*self.n_lo/(10**(-0.4*vegaMag))
            sig_m_hi = self.a_hi + 1.0857*self.n_hi/(10**(-0.4*vegaMag))
            lo = np.abs(np.random.normal(scale=1.0*(sig_m-sig_m_lo),size=s))
            hi = np.abs(np.random.normal(scale=1.0*(sig_m_hi-sig_m),size=s))
            x = np.random.random(size=s)
            sig_m += np.choose(x<0.5,[hi,-lo])
        except:
            pass
        sig_m = np.clip(sig_m,0.02,np.inf)
        return sig_m * f_nmgy / 1.0857

# unWISE photometric uncertainty functions (added April 17th 2019 by JT
# Schindler)
# Compared to the other ones, these directly use the calculated fluxes in
# nanomaggies
# Disclaimer: These flux errors are based on unWISE Stripe 82 data and
# might not be useful for the galactic plane

# AB mag version
_unwise_phot_pars = {
    'a':{'W1':0.20206436, 'W2': 4.35351555e-01},
    'b':{'W1': 0.11505964, 'W2': 9.35033234e-02},
    'c':{'W1': 0.1444974, 'W2': 4.83102906e-01},
    'd':{'W1': 12.98079923, 'W2': 1.00483359e+02},
    'a_lo':{'W1': 0.20370591, 'W2':  0.4236139},
    'b_lo':{'W1': 0.09542969, 'W2': 0.0972767},
    'c_lo':{'W1': 0.20835097, 'W2': 0.44459558},
    'd_lo':{'W1': 16.5664579, 'W2': 83.41207598},
    'a_hi':{'W1': 0.20433164, 'W2': 4.13220515e-01},
    'b_hi':{'W1': 0.09899131, 'W2': 9.90261430e-02},
    'c_hi':{'W1': 0.19675113, 'W2': 4.41065697e-01},
    'd_hi':{'W1': 20.31254652, 'W2': 1.06579650e+02},
    'ABtoVega':{'W1':2.699,'W2':3.339,'W3':5.174,'W4':6.620},
}



def phot_err_func(x, a, b, c, d):
    return a + b * np.sqrt(x * c + d)


class unwisePhotoUnc(object):
    def __init__(self, b):
        self.band = b
        self.a = _unwise_phot_pars['a'][b]
        self.b = _unwise_phot_pars['b'][b]
        self.c = _unwise_phot_pars['c'][b]
        self.d = _unwise_phot_pars['d'][b]
        self.a_lo = _unwise_phot_pars['a_lo'][b]
        self.b_lo = _unwise_phot_pars['b_lo'][b]
        self.c_lo = _unwise_phot_pars['c_lo'][b]
        self.d_lo = _unwise_phot_pars['d_lo'][b]
        self.a_hi = _unwise_phot_pars['a_hi'][b]
        self.b_hi = _unwise_phot_pars['b_hi'][b]
        self.c_hi = _unwise_phot_pars['c_hi'][b]
        self.d_hi = _unwise_phot_pars['d_hi'][b]
        self.vegaConv = _unwise_phot_pars['ABtoVega'][b]

    def __call__(self, f_nmgy):
        s = f_nmgy.shape

        # convert f_nmgy(ab) to f_nmgy(vega)
        f_nmgy = f_nmgy * 10 ** (-0.4 * (-self.vegaConv))

        # fit to median flux error
        sig_f = phot_err_func(f_nmgy, self.a, self.b, self.c, self.d)

        # calculating lower and upper 1-sigma ranges on the flux error
        sig_f_lo = phot_err_func(f_nmgy, self.a_lo, self.b_lo, self.c_lo,
                                 self.d_lo)
        sig_f_hi = phot_err_func(f_nmgy, self.a_hi, self.b_hi, self.c_hi,
                                 self.d_hi)
        # sampling the lower and upper flux error distributions
        lo = np.abs(np.random.normal(scale=1.0 * (sig_f - sig_f_lo), size=s))
        hi = np.abs(np.random.normal(scale=1.0 * (sig_f_hi - sig_f), size=s))
        # Randomly (with 50% probability for upper and lower)
        # assigning a width to the median flux error
        x = np.random.random(size=s)
        sig_f += np.choose(x < 0.5, [hi, -lo])

        # returns the flux error according to the flux in nanomaggies
        return sig_f


_vhs_phot_pars = {
    'a': {'J': 3.91643235e-02, 'K': 3.07582631e-02},
    'a_lo': {'J': 3.59169233e-02, 'K': 2.58237170e-02, },
    'a_hi': {'J': 3.97970589e-02, 'K': 3.97046265e-02, },
    'b': {'J': 8.16645695e-01, 'K': 8.00970418e-01},
    'b_lo': {'J': 8.33909820e-01, 'K': 1.01595882e+00, },
    'b_hi': {'J': 1.19999131e+00, 'K': 5.47209268e-01, },
    'c': {'J': 8.12484898e+01, 'K': 8.21096486e+01},
    'c_lo': {'J': 5.80885209e+01, 'K': 8.56731327e+01, },
    'c_hi': {'J': 1.28026269e+02, 'K': 7.49011435e+01, },

    'ABtoVega': {'J': 0.916, 'K': 1.827},
}


def phot_err_func_vhs(x, b, c, d):
    return b * np.sqrt(x * c + d)


class vhsPhotoUnc(object):
    """ The VHS photometry is somewhat inhomogeneous in its error properties.
     Thus our error model cannot adequately represent the different depths.
      J(AB) < 14 : errors are underestimated
      J(AB) > 22.5 : errors are overestimated
      J(AB) > 21.5 : error spread is too broad

      K(AB) > 22.5 : errors are overestimated

      (added on May 7th by JT Schindler)"""

    def __init__(self, b):
        self.band = b
        self.a = _vhs_phot_pars['a'][b]
        self.a_lo = _vhs_phot_pars['a_lo'][b]
        self.a_hi = _vhs_phot_pars['a_hi'][b]
        self.b = _vhs_phot_pars['b'][b]
        self.b_lo = _vhs_phot_pars['b_lo'][b]
        self.b_hi = _vhs_phot_pars['b_hi'][b]
        self.c = _vhs_phot_pars['c'][b]
        self.c_lo = _vhs_phot_pars['c_lo'][b]
        self.c_hi = _vhs_phot_pars['c_hi'][b]
        self.vegaConv = _vhs_phot_pars['ABtoVega'][b]

    def __call__(self, f_nmgy):
        s = f_nmgy.shape

        # fit to median flux error
        sig_f = phot_err_func_vhs(f_nmgy, self.a, self.b, self.c)
        try:
            # calculating lower and upper 1-sigma ranges on the flux error
            sig_f_lo = phot_err_func_vhs(f_nmgy, self.a_lo, self.b_lo, self.c_lo)
            sig_f_hi = phot_err_func_vhs(f_nmgy, self.a_hi, self.b_hi, self.c_hi)
            # sampling the lower and upper flux error distributions
            lo = np.abs(np.random.normal(scale=1.0 * (sig_f - sig_f_lo), size=s))
            hi = np.abs(np.random.normal(scale=1.0 * (sig_f_hi - sig_f), size=s))
            # Randomly (with 50% probability for upper and lower)
            # assigning a width to the median flux error
            x = np.random.random(size=s)
            sig_f += np.choose(x < 0.5, [hi, -lo])
        except ValueError:
            pass
        # returns the flux error according to the flux in nanomaggies
        return sig_f


def phot_err_func_des(x, b, c, d):
    return b * np.sqrt(x * c + d)

_des_phot_pars = {
  'a':{'y':3.31680680e-02, 'z':7.09471586e-03},
  'a_lo':{'y':4.20979994e-02, 'z':0.02960503 },
  'a_hi':{'y':3.97028685e-02, 'z':1.29752830e-02, },
  'b':{'y':3.97499533e-01, 'z':2.11597580e+00},
  'b_lo':{'y':2.24561584e-01, 'z':0.09037421 },
  'b_hi':{'y':3.00686511e-01, 'z':1.39335946e+00 },
  'c':{'y':3.03458372e+02, 'z':6.89729357e+02},
  'c_lo':{'y':1.15632183e+02, 'z':24.70625448 },
  'c_hi':{'y':4.19397683e+02, 'z':4.37006613e+02},
}


class DESPhotoUnc(object):
    """ The DES error function for the z/Y band is an approximation to the
    real error properties. It somewhat underestimates the error between
    20-22.5 mag and overestimates the error above these values. It is
    accurate towards brighter magnitudes. (added May 7th 2019 by JT
    Schindler) """

    def __init__(self, b):
        self.band = b
        self.a = _des_phot_pars['a'][b]
        self.a_lo = _des_phot_pars['a_lo'][b]
        self.a_hi = _des_phot_pars['a_hi'][b]
        self.b = _des_phot_pars['b'][b]
        self.b_lo = _des_phot_pars['b_lo'][b]
        self.b_hi = _des_phot_pars['b_hi'][b]
        self.c = _des_phot_pars['c'][b]
        self.c_lo = _des_phot_pars['c_lo'][b]
        self.c_hi = _des_phot_pars['c_hi'][b]

    def __call__(self, f_nmgy):
        s = f_nmgy.shape

        # fit to median flux error
        sig_f = phot_err_func_des(f_nmgy, self.a, self.b, self.c)

        try:
            # calculating lower and upper 1-sigma ranges on the flux error
            sig_f_lo = phot_err_func_des(f_nmgy, self.a_lo, self.b_lo, self.c_lo)
            sig_f_hi = phot_err_func_des(f_nmgy, self.a_hi, self.b_hi, self.c_hi)
            # sampling the lower and upper flux error distributions
            lo = np.abs(np.random.normal(scale=1.0 * (sig_f - sig_f_lo), size=s))
            hi = np.abs(np.random.normal(scale=1.0 * (sig_f_hi - sig_f), size=s))
            # Randomly (with 50% probability for upper and lower)
            # assigning a width to the median flux error
            x = np.random.random(size=s)
            sig_f += np.choose(x < 0.5, [hi, -lo])
        except ValueError:
            pass
        # returns the flux error according to the flux in nanomaggies
        return sig_f

_tmass_phot_pars = {
    'x0':{'J':14.47706168, 'H':13.74112897,
     'K':13.10023926 },
    'y0':{'J':-3.56813635, 'H':-3.51598265,
     'K':-3.60406916 },
    'k':{'J':0.69856564, 'H':0.75697424,
         'K':0.79939991 },
    'c':{'J':0.21028606, 'H':0.30856437,
         'K':0.28210028 },
    'ABtoVega':{'J':0.894,'H':1.374,'K':1.84}
}

def piecewise_linear(x, x0, y0, k):
    return np.piecewise(x, [x < x0], [lambda x: y0, lambda x:k*x + y0-k*x0])


class tmassPhotoUnc(object):
    def __init__(self,b):
        self.band = b
        self.x0 = _tmass_phot_pars['x0'][b]
        self.y0 = _tmass_phot_pars['y0'][b]
        self.k = _tmass_phot_pars['k'][b]
        self.c = _tmass_phot_pars['c'][b]
        self.vegaConv = _tmass_phot_pars['ABtoVega'][b]

    def __call__(self,f_nmgy):
        abMag = nmgy2abmag(self.band,f_nmgy)
        vegaMag = abMag - self.vegaConv
        sig_m = piecewise_linear(vegaMag,self.x0,self.y0,self.k)
        s = f_nmgy.shape
        sig_m_scatter = np.random.normal(scale=0.5*self.c,size=s)
        sig_m +=sig_m_scatter
        # sig_flux = np.exp(sig_m)
        # sig_flux = (-0.4*np.log(10) * sig_m * np.power(10,-0.4*abMag) * 3631)
        sig_m = np.exp(sig_m)
        return sig_m * f_nmgy /1.0857


# Included PanSTARRS photometric uncertainty (Jinyi Yang, Feige Wang)

 # PS1 photometric model

_ps1_phot_pars = {
  'n':{'g':3.03067231e-10, 'r':3.51954013e-10,'i':4.02844834e-10, 'z':7.21769276e-10, 'y':1.75281748e-09 },
  'n_lo':{'g':2.64619046e-10, 'r':3.10547692e-10,'i':3.56529614e-10,'z':6.20530326e-10,'y':1.52490945e-09},
  'n_hi':{'g':3.41515417e-10, 'r':3.93360334e-10,'i':4.49160054e-10,'z':8.23008227e-10,'y':1.98072551e-09},
  'a':{'g':3.12976920e-03, 'r':2.64537399e-03,'i':2.47470270e-03, 'z':3.00356517e-03,'y':3.69861972e-03},
  'a_lo':{'g':3.00755710e-03, 'r':2.51321863e-03,'i':2.37777787e-03,'z':2.96226831e-03,'y':3.62367433e-03},
  'a_hi':{'g':3.25198130e-03, 'r':2.77752935e-03,'i':2.57162754e-03,'z':3.04486204e-03,'y':3.77356511e-03},
}


class ps1PhotoUnc(object):

    def __init__(self,b):

        self.band = b
        self.n = _ps1_phot_pars['n'][b]
        self.n_lo = _ps1_phot_pars['n_lo'][b]
        self.n_hi = _ps1_phot_pars['n_hi'][b]
        self.a = _ps1_phot_pars['a'][b]
        self.a_lo = _ps1_phot_pars['a_lo'][b]
        self.a_hi = _ps1_phot_pars['a_hi'][b]

    def __call__(self,f_nmgy):

        ABMag = nmgy2abmag(self.band,f_nmgy)
        s = f_nmgy.shape
        sig_m = 1.0857*self.n/(10**(-0.4*ABMag)) #/ np.sqrt(np.random.randint(1,5,size=s))

        try:
            # this is legacy code for approximating the depth variations
            # over the sky, not really sure how valid it is
            sig_m_lo = 1.0857*self.n_lo/(10**(-0.4*ABMag)) #/ np.sqrt(np.random.randint(1,5,size=s))
            sig_m_hi = 1.0857*self.n_hi/(10**(-0.4*ABMag)) #/ np.sqrt(np.random.randint(1,5,size=s))

            lo = np.abs(np.random.normal(scale=1.0*(sig_m-sig_m_lo),size=s))
            hi = np.abs(np.random.normal(scale=1.0*(sig_m_hi-sig_m),size=s))

            #
            lo = np.abs(np.random.normal(scale=np.max([np.abs(sig_m-sig_m_lo),1e-3]),size=s))
            #
            hi = np.abs(np.random.normal(scale=np.max([np.abs(sig_m_hi-sig_m),1e-3]),size=s))

            x = np.random.random(size=s)

            sig_m += np.choose(x<0.5,[hi,-lo])

        except:

            pass

        sig_m = np.clip(sig_m,0.0,np.inf) #/np.sqrt(np.random.randint(1,6,size=s))

        return sig_m * f_nmgy / 1.0857




supported_photo_systems = {
  'SDSS':{
    'Legacy':{'bands':'ugriz','magSys':'asinh','uncMap':sdssPhotoUnc},
    'Stripe82':{'bands':'ugriz','magSys':'AB','uncMap':sdssStripe82PhotoUnc},
  },
  'CFHT':{
    'CFHTLS_Wide':{'bands':'ugriz','magSys':'AB','uncMap':cfhtlsWidePhotoUnc},
  },
  'UKIRT':{
    'UKIDSS_LAS':{'bands':'YJHK','magSys':'AB','uncMap':ukidsslasPhotoUnc},
    'UKIDSS_DXS':{'bands':'JHK','magSys':'AB','uncMap':ukidssdxsPhotoUnc},
  },
  'WISE':{
    'AllWISE':{'bands':['W1','W2'],'magSys':'AB','uncMap':allwisePhotoUnc},
    'unWISE':{'bands':['W1','W2'],'magSys':'AB','uncMap':unwisePhotoUnc},
  },
  'TMASS':{
    'Allsky':{'bands':['J','H','K'],'magSys':'AB','uncMap':tmassPhotoUnc},
  },
  'DECam':{
    'DECaLS':{'bands':'grz','magSys':'AB','uncMap':None},
    'DES':{'bands':'grizy','magSys':'AB','uncMap':None},
    'DES_err':{'bands':'zy','magSys':'AB','uncMap':DESPhotoUnc},
  },
  'HSC':{
    'Wide':{'bands':'grizy','magSys':'AB','uncMap':None},
  },
  'LSST':{
    'Wide':{'bands':'ugrizy','magSys':'AB','uncMap':None},
  },
  'PanSTARRS':{
    'PS1':{'bands':'grizy','magSys':'AB','uncMap':ps1PhotoUnc}
  },
  'GaiaMA18': {
    'Gaia': {'bands': ['G', 'GBPb', 'GBPf', 'GRP'], 'magSys': 'AB',
             'uncMap': None}
  },
  'VISTA':{
    'VISTA':{'bands':['Z','Y','J','H','K'],'magSys':'AB',
             'uncMap':None},
    'VHS':{'bands':['J','K'],'magSys':'AB',
             'uncMap':vhsPhotoUnc},
  },
    'Euclid':{
        'Euclid':{'bands':['V','Y','J','H'],'magSys':'AB',
             'uncMap': None}
    }
}

# should find a better container / organization for this
def load_photo_map(photSystems):
    bandpasses = OrderedDict()
    filterdata = fits.open(os.path.join(datadir, 'filtercurves.fits'))
    mapObserved = {}
    magSys = {}
    filtName = {} # ugh
    for photDesc in photSystems:
        try:
            photSysName,survey,bands = photDesc
        except ValueError:
            photSysName,survey = photDesc
            bands = None
        try:
            photSys = supported_photo_systems[photSysName][survey]
        except:
            raise ValueError('%s-%s not a valid photo system' %
                             (photSysName,survey))
        if bands is None:
            bands = photSys['bands']
        for band in bands:
            bpName = '-'.join([photSysName,survey,band])
            # a workaround for the naming of the extension in the filter file
            _photSysName = {'UKIRT':'UKIDSS'}.get(photSysName,photSysName)
            bpExt = '-'.join([_photSysName,band])
            fdat = filterdata[bpExt].data
            fcurv = interp1d(fdat.lam.astype(np.float64),
                             fdat.Rlam.astype(np.float64),
                             bounds_error=False,fill_value=0.0,kind='slinear')
            # precompute the bandpass normalization
            norm = simps(fdat.Rlam/fdat.lam, fdat.lam)
            bandpasses[bpName] = dict(Rlam=fcurv,norm=norm,data=fdat)
            if photSys['uncMap'] is not None:
                mapObserved[bpName] = photSys['uncMap'](band)
            magSys[bpName] = photSys['magSys']
            filtName[bpName] = bpExt

    print (mapObserved)

    return dict(bandpasses=bandpasses,mapObserved=mapObserved,
                magSys=magSys,filtName=filtName)

def getPhotoCache(wave,photoMap):
    photoCache = OrderedDict()
    for b,bp in photoMap['bandpasses'].items():
        bpdata = bp['data']
        i1,i2 = np.searchsorted(wave,bpdata['lam'][[0,-1]],side='right')
        if i1==i2:
            lam,Rlam,dlam = 0.0,0.0,0.0
        else:
            lam = wave[i1:i2]
            dlam = np.diff(wave[i1:i2])
            dlam = np.concatenate([dlam,[dlam[-1],]])
            Rlam = bp['Rlam'](wave[i1:i2])
        photoCache[b] = {'ii':(i1,i2),'lam_Rlam_dlam':lam*Rlam*dlam,
                         'norm':bp['norm']}
    return photoCache

conv_Slam_to_Snu = 1/(c_Angs * 3631e-23)

def calcSynPhot(spec,photoMap=None,photoCache=None,mags=None,fluxes=None):
    if photoCache is None:
        photoCache = getPhotoCache(spec.wave,photoMap)
    if mags is None:
        mags = np.zeros(len(photoCache))
    if fluxes is None:
        fluxes = np.zeros(len(photoCache))
    for j,b in enumerate(photoCache):
        fnorm = photoCache[b]['norm']
        i1,i2 = photoCache[b]['ii']
        lamRlamdlam = photoCache[b]['lam_Rlam_dlam']
        flam = spec.f_lambda[i1:i2]
        flux = np.sum(flam*lamRlamdlam) / fnorm
        fluxes[j] = flux * conv_Slam_to_Snu
        if fluxes[j] == 0:
            mags[j] = 99.99
        else:
            mags[j] = min(-2.5*np.log10(fluxes[j]),99.99)  # AB mag
    fluxes *= 1e9 # nanomaggies
    return mags,fluxes

def calcObsPhot(synFlux,photoMap,seed=None):
    obsFlux = np.empty_like(synFlux)
    obsFluxErr = np.empty_like(synFlux)
    obsMag = np.empty_like(synFlux)
    obsMagErr = np.empty_like(synFlux)
    gridShape = synFlux.shape[:-1]
    if seed:
        np.random.seed(seed)
    for j,b in enumerate(photoMap['bandpasses']):
        _b = b.split("-")[-1] # the short filter name
        obsFluxErr[...,j] = photoMap['mapObserved'][b](synFlux[...,j])
        obsFlux[...,j] = synFlux[...,j] + \
                           obsFluxErr[...,j]*np.random.randn(*gridShape)
        if photoMap['magSys'][b]=='AB':
            obsMag[...,j],obsMagErr[...,j] = nmgy2abmag(_b,obsFlux[...,j],
                                                        obsFluxErr[...,j])
        elif photoMap['magSys'][b]=='asinh':
            obsMag[...,j],obsMagErr[...,j] = nmgy2asinhmag(_b,obsFlux[...,j],
                                                           obsFluxErr[...,j])
        else:
            raise ValueError
    return Table({'obsFlux':obsFlux,'obsFluxErr':obsFluxErr,
                  'obsMag':obsMag,'obsMagErr':obsMagErr})

class LazyPhotoMap(object):
    '''Needed for using photoMap in multiprocessing calls'''
    def __init__(self,photSystems):
        self.photSystems = photSystems
        self.photoMap = None
        self.photoCache = None
    def __loadup(self,spec=None):
        if self.photoMap is None:
            self.photoMap = load_photo_map(self.photSystems)
        if self.photoCache is None and spec is not None:
            self.photoCache = getPhotoCache(spec.wave,self.photoMap)
    def calcSynPhot(self,spec):
        self.__loadup(spec)
        return calcSynPhot(spec,photoCache=self.photoCache)
    def getBandpasses(self):
        self.__loadup()
        return self.photoMap['bandpasses']
