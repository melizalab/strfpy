# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""estimate spectrotemporal receptive fields
Copyright (C) 2016 Tyler Robbins
"""

from __future__ import (division, print_function, absolute_import)

import numpy as np
import scipy.ndimage.filters as sf
import utils
import emcee
import time

def reverse_correlation(stims,psths,width,normalize=None,ridge_param=5e-4,eig_cutoff=1e-4,smooth=0,rescale=True):
    """Calculate spike triggered average (STA) using reverse correlation 

    stims       : array of spectrograms
    psths       : array of peristimulus time histograms
    width       : number linear filter time steps  
    normalize   : method for normalizing the stimulus covariance matrix [None, ridge, pseudo_inverse, inverse]
    ridge_param : amount of regularization when using ridge regression
    eig_cutoff  : variance ratio under which eigenvalues will be ignored when using pseudo_inverse  
    rescale     : scales maximum value of returned STA to one

    """
    nstim = len(stims)

    STIM = np.concatenate([s - np.mean(s) for s in stims],1)
    R = np.concatenate([p - np.mean(p) for p in psths])

    sres, dsdur = STIM.shape
    STA = np.zeros((sres,int(dsdur)))


    XX = np.asarray(STIM)

    if normalize != None:
        covXX = np.dot(XX,XX.T)

        if normalize == "ridge":
            reg = ridge_param*np.identity(sres)
            nvcovXX = np.linalg.inv(covXX) + reg

        if normalize == "pseudo_inverse":
            U,s,V = np.linalg.svd(covXX)
            invs = 1/s
            varexpl = s**2/dsdur
            varratio = varexpl/np.sum(varexpl)
            invs[varratio<eig_cutoff] = 0
            nvcovXX = np.dot(V.T,np.dot(np.diag(invs),U.T))

        if normalize == "inverse":
            nvcovXX = np.linalg.inv(covXX)

        XX = np.dot(nvcovXX,XX)

    for i in range(sres):
        STA[i,::-1] = np.correlate(XX[i,:],R,mode="same")

    STA = STA[:,int(round(dsdur/2-width/2-1)):int(round(dsdur/2+width/2-1))]
    if smooth: STA = sf.gaussian_filter(STA,smooth)
    if rescale: STA /= abs(STA).max()
    return STA

# hack to avoid multiprocessing pickling errors
# https://github.com/dfm/emcee/issues/148
def _extract_cost(theta, name, *args):
    hack = eval(name)
    return hack._cost(theta, *args)

class _fit_strf:
    def __init__(self, stims, psth_data,tres=15):
        self.stims = stims
        self.psth_data = psth_data
        self.tres = tres
        self.STA = reverse_correlation(stims,psth_data,tres,normalize="pseudo_inverse",smooth=1)
        self.Sfact, self.Tfact = utils.factorize(self.STA)
        self.channels, self.sres = self.Sfact.shape

    @staticmethod
    def _cost(theta):
        return 0

    def _pos0(nwalkers):
        return np.zeros(nwalkers)

    def fit(self, burn=1000, smpl=1, nwalkers=200, ret_flatchains=False, threads=None, pool=None):

        pos = self._pos0(nwalkers)

        ndim = np.shape(pos)[1]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, _extract_cost, 
                                        args=[self.__class__.__name__,self.stims,self.psth_data,self.sres,self.tres,self.channels],
                                        threads=threads)
        print("Sampling...")
        begin = time.time()

        # burn in the chains
        pos2, prob, state = sampler.run_mcmc(pos, burn, storechain=False)

        # sample the posterior distribution of the parameters.
        sampler.reset()
        pos3, prob2, state2 = sampler.run_mcmc(pos2, smpl, storechain=True)

        end = time.time()
        finish_time = time.strftime("%Y%m%d-%H%M",time.localtime())

        wait_duration = (end - begin)/60

        print("Sampling finished at " + finish_time + ". Took %f minutes." % wait_duration)

        self._results(sampler)

class emcee_factorized(_fit_strf):
    @staticmethod
    def _cost(theta, stims, psth_data, sres, tres, channels):
        # bounds on the uniform prior distributions 
        if not (max(abs(theta)) <= 1): return -np.inf

        Tfilt = theta[:tres*channels]
        Sfilt = theta[tres*channels:]

        Tfilt = np.reshape(np.mat(Tfilt),(channels,tres))
        Sfilt = np.reshape(np.mat(Sfilt),(channels,sres))
        nh = np.asarray(Sfilt.T*Tfilt)

        out = 0
        for i,stim in enumerate(stims):
            rate = utils.speccnov(nh,stim)
            out += np.sum((utils.normalize(psth_data[i]) - utils.normalize(rate))**2)/len(psth_data[i])
        return  -out

    def _pos0(self, nwalkers, sigma=0.1):
        Tfilt_start = emcee.utils.sample_ball(self.Tfact.flatten(),[sigma]*self.tres*self.channels,size=nwalkers)
        Sfilt_start = emcee.utils.sample_ball(self.Sfact.flatten(),[sigma]*self.sres*self.channels,size=nwalkers)

        pos = np.concatenate((Tfilt_start,Sfilt_start),1)
        return pos

    def _results(self,sampler):
        best = sampler.flatchain[sampler.flatlnprobability.argmax()]
        newTfilt = best[:self.tres*self.channels]
        newSfilt = best[self.tres*self.channels:]
        newTfilt = np.reshape(np.mat(newTfilt),(self.channels,self.tres))
        newSfilt = np.reshape(np.mat(newSfilt),(self.channels,self.sres))
        newstrf = np.asarray(newSfilt.T*newTfilt)
        self.sampler = sampler
        self.maxlik = (newstrf,newSfilt,newTfilt)

class emcee_parameterized(_fit_strf):
    @staticmethod
    def _cost(theta, stims, psth_data, sres, tres, channels):
        tres = int(np.rint(theta[0]))
        theta = theta[1:].reshape(channels,7)

        for i in range(channels):
            z1, p1, p2, p3, l, mu, sig = theta[i]    
            # bounds on the uniform prior distributions 
            if not (   
                  5 < tres <= 150 and
                 -100 < z1  < 100 and   
                 -100 < p1  < p2 < p3 < 100 and
                 0 < l  < 50 and
                 0 < mu  < sres and
                 0 < sig < sres
                ): return -np.inf

        Sfilt = np.empty((channels,sres))
        Tfilt = np.empty((channels,tres))

        for i in range(channels):
            z1,p1,p2,p3,l, mu, sig = theta[i]
            Tfilt[i] = np.asarray(utils.overtime(0,tres,utils.P3Z1,z1,p1,p2,p3,l,1))
            Sfilt[i] = utils.gauss(np.arange(0,sres),mu,sig)

        nh = np.matmul(Sfilt.T,Tfilt)

        out = 0
        ntrials = np.size(psth_data)
        for i,stim in enumerate(stims):
            rate = utils.speccnov(nh,stim)
            out += np.sum((utils.normalize(psth_data[i]) - utils.normalize(rate))**2)/len(psth_data[i])
        return  -out

    def _pos0(self, nwalkers, sigma=0.1):
        pos = np.concatenate((   
              np.random.uniform(-100, 100, (nwalkers,1)),
              np.random.uniform(-100, -50, (nwalkers,1)),
              np.random.uniform(-50, 50, (nwalkers,1)),
              np.random.uniform( 50, 100, (nwalkers,1)),
              np.random.uniform(   0,   1, (nwalkers,1)),
              np.random.uniform(   0,  15, (nwalkers,1)),
              np.random.uniform(   0,  15, (nwalkers,1))
              ),1)

        pos = emcee.utils.sample_ball([15]+[0,-75,0,75,0.5,7,7]*self.channels,[0.1]+[0.1]*self.channels*7,size=nwalkers)

        return pos

    def _results(self,sampler):
            best = sampler.flatchain[sampler.flatlnprobability.argmax()]
            tres = int(np.rint(best[0]))
            best = best[1:].reshape(self.channels,7)
            newSfilt = np.empty((self.channels,self.sres))
            newTfilt = np.empty((self.channels,tres))

            for i in range(self.channels):
                z1,p1,p2,p3,l, mu, sig = best[i]
                newTfilt[i] = np.asarray(utils.overtime(0,tres,utils.P3Z1,z1,p1,p2,p3,l,1))
                newSfilt[i] = utils.gauss(np.arange(0,self.sres),mu,sig)

            newstrf = np.dot(newSfilt.T,newTfilt)
            newstrf /= np.max(np.abs(newstrf))
            self.sampler = sampler
            self.maxlik = (newstrf,newSfilt,newTfilt)
