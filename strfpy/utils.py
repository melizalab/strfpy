from __future__ import (division, print_function, absolute_import)

import numpy as np
import gammatone.gtgram as gg
import libtfr

from scipy.signal import decimate, resample, resample_poly
from scipy.io.wavfile import read 


def normalize(x):
    return (x-np.mean(x))/np.std(x)

def specconv(h,s):
    npix, nts = np.shape(s)
    cs = np.zeros(nts)
    for i in range(npix):
        cs += np.convolve(s[i,:],h[i,:],'same')
    return cs 

def generate_strf(resolution=50,time=50,maxfreq=8000,latency=0,frequency=0,A=0.25,sigma=0.1,gamma=0.001,alpha=1.4,beta=1.5):
    time -= 1
    scale = resolution/50.0   
    t = np.arange(float(np.negative(time)),1)
    tscale = np.arange(np.negative(time),1,2)
    x = latency
    f = np.arange(0,maxfreq+1,float(maxfreq)/resolution)
    y = frequency
    tc = t+x
    fc = f-y
    tprime, fprime = np.meshgrid(tc,fc)
    sigma = sigma/scale
    Gtf = A*np.exp(-sigma**2*tprime**2-gamma**2*fprime**2)*(1-alpha**2*sigma**2*tprime**2)*(1-beta**2*gamma**2*fprime**2)
    return (Gtf,tscale,f)

def evaluate(h,stims,rs):
    corcof = 0
    nstim = len(stims)
    for i, stim in enumerate(stims):
        R = speccnov(np.asarray(h),stim)
        R += np.mean(rs[i])
        corcof += np.corrcoef(rs[i],R)[0][1]
    return corcof/nstim

def load_sound_stims(files,root="",windowtime=0.016,ovl=0.0016,f_min=500,f_max=8000,gammatone=False,
                    dsample=10,sres=15,compress=0):
    stims = []
    durations = []

    for f in files:
        Fs, wave = read(root+f)
        duration = int(1000*len(wave)/Fs)
        durations.append(duration)
        if gammatone:
            Pxx = gg.gtgram(wave,Fs,windowtime,ovl,sres,f_min)
            Pxx = np.log10(Pxx)

        else:
            w = np.hanning(int(windowtime*Fs))
            Pxx = libtfr.stft(wave, w, int(w.size * .1))
            freqs, ind = libtfr.fgrid(Fs, w.size, [f_min, f_max])
            Pxx = Pxx[ind,:]
            Pxx = np.log10(Pxx+compress)
            Pxx = resample(Pxx,sres)
        Pxx = resample(Pxx,duration/dsample,axis=1)
        stims.append(Pxx)
    return stims,durations

def prune_walkers(pos,lnprob,tolerance=10,resample=None,return_indx=False,cutoff_exclude=10):
    mean_lnprob = np.mean(lnprob,axis=1)
    sorted_mean = np.sort(mean_lnprob)
    gradient = np.gradient(sorted_mean)[:-cutoff_exclude]
    
    cutoff_indx = np.where(gradient>np.mean(gradient)*tolerance)[0][-1]
    cutoff = sorted_mean[cutoff_indx]
    prune = np.where(mean_lnprob > cutoff)[0]
   
    if resample: prune = np.random.choice(prune,resample)
    if return_indx: return pos[prune], prune
    else: return pos[prune] 
    
def P3Z1(t,z1,p1,p2,p3,l,A):
    zeros = (t-z1)
    poles = (t+p1)*(t+p2)*(t+p3)
    out = A*np.exp(-l*t)*poles/zeros
    return out if np.isfinite(out) else 0 
    
def PZ(t,zs,ps,l,A):
    zeros = 1
    poles = 1
    for z in zs: zeros *= (t-z)
    for p in ps: poles *= (t+p)
    out = A*np.exp(-l*t)*poles/zeros
    return out if np.isfinite(out) else 0 

def gauss(x, mu=0, sig=1):
    return np.exp(-np.power((x-mu),2)/(2*np.power(sig,2)))/np.sqrt(2*np.power(sig,2)*np.pi)

def morlet(x, mu=0, sig=1, dep=1):
    return np.exp(-np.power((x-mu),2)/(np.power(sig,2)) - 1j*dep*(x-mu))

def overtime(l, u, f, *args):
    out = []
    for a in range(l,u):
        out.append(f(a,*args))
    return out

def factorize(strf,channels=None):
    sres, tres = np.shape(strf)
    U,s,V = np.linalg.svd(strf)
    
    if not channels: channels = len(np.where(s>1.0)[0]) 
    
    time = np.ndarray((channels,tres))
    spec = np.ndarray((channels,sres))
    for i in range(channels):
        time[i] = V[i,:]*s[i]/max(s)
        spec[i] = U[:,i]
    
    return spec, time

def normalized_cross_correlation(a,b):
    xcorr = np.correlate(a,b)
    acorr = np.correlate(a,a)
    bcorr = np.correlate(b,b)
    
    return xcorr/(np.sqrt(acorr)*np.sqrt(bcorr))