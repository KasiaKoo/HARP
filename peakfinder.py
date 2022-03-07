# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:35:28 2020

@author: Timur
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from scipy.special import erf as erf
import multiprocessing as mp
import scipy.optimize as op
from IPython import embed


"""
Ok here's the usage for this:
    1) Smooth(curve)
    2) fitgaussians(curve)
        returned set: Peaks[positions,peakvalues],sigmas,
    3) If you want to check how good the fitting is:
    
"""

def getpeaks(b,threshold=0.1,absval = None):
    c=np.gradient(b)
    d=np.gradient(np.gradient(b))
    miny=threshold*absval if absval else np.amax(b)*threshold
    e = ((c*np.roll(c,1))< 0) * (d<0) * (b>miny)
    f = np.array(np.where(e==True))

    return np.vstack((f[0], b[f]))


def smooth(curve,shift=5):
    c = np.concatenate((curve[:5],((curve+np.roll(curve,shift)+np.roll(curve,-shift))/3)[5:-5],curve[-5:]))
    return c

def get2dbg(image,margin=5):
    ymax=len(image)
    xmax=len(image[0])
    
    lims=np.ndarray((ymax,xmax,4))
    ylims = np.array([(i - margin if i>margin else 0,i + margin if i< ymax-margin else ymax) for i in range(ymax)])
    xlims = np.array([(i - margin if i>margin else 0,i + margin if i< xmax-margin else xmax) for i in range(xmax)])
    
    #can this be replaced with np.fromiter?
    lims = it.product(xlims,ylims)
    tlim = [np.sum(image[lim[1][0]:lim[1][1],lim[0][0]:lim[0][1]]) for lim in lims]
    rlim = (np.array(tlim).reshape((xmax,ymax))/pow(2*margin,2)).transpose()
    return rlim

gaus  = lambda x,x0,a,s,sk=0    : a*np.exp(pow(((x-x0)/s),2)/-2)*((1+erf(sk*(x-x0)/(np.sqrt(2)*s))))

def getsigma(curve, peaks, frac=0.9, delta=0.01):
    
    peaks=peaks.T

    sigs=np.empty((len(peaks)))
    for i,peak in enumerate(peaks):
        val    = peak[1]
        place  = peak[0]
        newval = frac*val

        s      = np.array(np.where( (newval - (val*delta) < curve)*(curve <newval + (val*delta)))).flatten()
        #embed()
        try:
            bot    = np.amax(s*(s<place))
            top    = np.amin(s[np.nonzero(s*(s>place))])
            sig1   = (place - bot) / np.sqrt(-2*np.log(frac))
            sig2   = (top - place) / np.sqrt(-2*np.log(frac))
            sigs[i]=min(sig1,sig2)
        except:
            sigs[i]=1
        
    return sigs


def reconstructgaussians(peaks,sigs,skews,axlen):
    recon = np.sum(np.array([gaus(np.arange(axlen),peaks[0,i],peaks[1,i],sigs[i],skews[i]) for i in range(len(peaks[0]))]),
                axis=0)
    return recon

def getdif(peaks,sigs,skews,curve):
    recon   = reconstructgaussians(peaks,sigs,skews,len(curve))
    dif     = curve - recon
    return dif

def getgausresidual(peaks,sigs,skews,curve):
    dif     = getdif(peaks,sigs,skews,curve)
    return np.sum(abs(dif))

def devolvetolist(peaks,sigs,skews):
    return np.vstack((peaks,sigs,skews)).T.flatten()

def reformarray(datlist):
    peaks,sigs,skews = np.split(datlist.reshape(int(len(datlist)/4),4).T,(2,3))
    return peaks,sigs.flatten(),skews.flatten()

def sqerror(peaks,sigs,skews,c):
    return pow(getdif(peaks,sigs,skews,c),2)

def sqewl(datlist,c) :
    peaks,sigs,skews = reformarray(datlist)
    return sqerror(peaks,sigs,skews,c)

def fitgaussians(curve,minpeak=0.1,sigheight = 0.8,sigstart=0.1,guesses=None,tol=2e-8):

    if not guesses:
        #Get the peaks and get sigmas
        peaks   = getpeaks(curve,minpeak)
        sigs    = getsigma(curve,peaks,sigheight,sigstart)
        skews    = np.zeros(len(sigs))
        #Find the initial errors in the reconstructed gaussians
        dif     = getdif(peaks,sigs,skews,curve)
        newpks  = getpeaks(dif,minpeak,np.amax(peaks[1]))
    else:
        peaks,sigs,skews = guesses
        dif     = getdif(peaks,sigs,skews,curve)
        newpks  = getpeaks(dif,minpeak,np.amax(peaks[1]))

    #Find any peaks that were hidden by larger features
    while len(newpks[0])>0:
        npk      = getpeaks(dif,0.1,np.amax(peaks[1]))
        nsig     = getsigma(dif,npk,sigheight,sigstart)
        peaks    = np.append(peaks,npk,axis=1)
        sigs     = np.append(sigs,nsig)
        skews    = np.zeros(len(sigs))
        dif      = getdif(peaks,sigs,skews,curve)
        newpks   = getpeaks(dif,0.1,np.amax(peaks[1]))

    residual  = getgausresidual(peaks,sigs,skews,curve)
    # print("Initial residual is ",residual, "over", len(peaks[0]),"peaks")

    ### New code 
    startvals = devolvetolist(peaks,sigs,skews)
    val, success = op.leastsq(sqewl,startvals[:len(curve)],args=curve,ftol=tol)
    peaks,sigs,skews = reformarray(val)
    # print("Final residual is {}".format(getgausresidual(peaks,sigs,skews,curve)))

    return peaks,sigs,skews

    

def plotresidual(peaks,sigs,skews,curve):
    recon = reconstructgaussians(peaks,sigs,skews,len(curve))
    plt.plot(curve)
    plt.plot(recon)
    plt.plot(getdif(peaks,sigs,skews,curve))
    plt.legend(("Curve","Reconstructed curve","Residuals"))   
    plt.show()


def randomgaussians(axis, maxgauss=10):
    numgaus = max(int(np.random.rand()*maxgauss),1)
    peaks   = (np.random.rand(2,numgaus).T*np.array((len(axis),6))).T
    sigs    = (np.random.rand(numgaus)*len(axis)/20)
    skews   = ((np.random.rand(numgaus)-0.5)*2)
    curve = reconstructgaussians(peaks,sigs,skews,len(axis))
    return peaks,sigs,skews,curve


if __name__=="__main__":
    axis    = np.linspace(-10,10,10000)
    spacing = axis[1]-axis[0]
    cats    = lambda x: gaus(x,-2,2,1) + gaus(x,3,1.5,1.5) + gaus(x,5,3,0.7)
    spikes  = lambda x: pow(np.sin(3*x) * np.sin(7.23476*x)/3 * np.sin(13.26843*x),2)/7
    beats   = lambda x: pow(np.sin(0.7*x) * np.sin(17.23476*x)/5 * np.sin(33.26843*x),2)/9
    testpeaks,testsigs,testskews,rands   = randomgaussians(axis,15)


    curve = rands


    start = time.time()
    peaks,sigs,skews = fitgaussians(curve,minpeak=0.1,sigheight=0.9,sigstart=0.01,tol=1e-2)
    finish = time.time() - start
    print("Time taken was {} seconds".format(finish))
    recon = reconstructgaussians(peaks,sigs,skews,len(curve))
    plt.plot(curve)
    plt.plot(recon)
    plt.plot(getdif(peaks,sigs,skews,curve))
    plt.legend(("Curve","Reconstructed curve","Residuals"))
    plt.show()
    print("Peaks found at: {}, heights are {}".format(axis[np.vectorize(int)(peaks[0])],peaks[1]))
