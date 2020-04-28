
import matplotlib.pyplot as plt
import numpy
import matplotlib
import matplotlib.colors as colors
#from partials import zintegral_fast, restrate_Ia, sigOM0sqinv
from matplotlib import ticker, cm
from astropy.cosmology import FlatLambdaCDM
import scipy.integrate as integrate
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import sys
#sys.path.insert(1, '/Users/akim/project/PeculiarVelocity/doc/src/')

from partials import zintegral_fast, restrate_Ia, sigOM0sqinv

# AP March 2020
# GW 3G
# This code is useful to run jobs in parallel

#Turn mpi on to run in parallel
mpi = True 
if mpi: 
    from joblib import Parallel, delayed
    njobs = 40

skyfrac=0.9 
# Perhaps skyfrac should be ~0.9 because we may not be able to see counterparts behind the plane
zmax=0.3
skyarea = skyfrac*4*numpy.pi
restrate_BNS = 1090/1e9/(0.679)**3  # best estimate from GW190425 in h=1 units, 90% limits are 2080- 350/1e9 # Mpc^-3 yr^-1  
#Decide if you want to get results for different zmax or durations
do_zmax_matrix=True
do_duration_matrix=False
#Want to use the galaxis in the overdensity power spectrum?
gsurvey=True


def N_yr(zmax,restrate):
    OmegaM0 = 0.28
    cosmo = FlatLambdaCDM(H0=100, Om0=OmegaM0)
    ans = integrate.quad(lambda x: cosmo.differential_comoving_volume(x).value/(1+x), 0, zmax)
    return(ans[0]*restrate*4*numpy.pi)

logdurations = numpy.arange(-0.4,1.5,1.2/8)
#logdurations = numpy.arange(0.4,1.4,1.2/8)
durations = 10**(logdurations)
zmaxs = numpy.arange(0.05,0.30001,0.25/10)
# Different uncertainties between 1-20% Remember this is ~2*sigma_d because it's sigma_mag
sigm_GWs = [0.005,0.01,0.02,0.05,0.1, 0.15, 0.2, 0.3] #at z=0.1  #numpy.arange(0.02,0.501,0.45/10)
X, Y = numpy.meshgrid(numpy.log10(durations*skyarea/4/numpy.pi*N_yr(zmax,restrate_BNS)), sigm_GWs)


var =[]

def fisher(zmax,duration,sigm,restrate,sigOM0sqinv,GW=True,gsurvey=gsurvey):
    print("Computing Fisher for ", duration," years, sigm=", sigm)
    f00,f11,f10, f02,f12,f22,lw0,lwa,bw0,bwa,Ow0,Owa,w0w0,w0wa,wawa = zintegral_fast(zmax,None,duration,sigm,restrate,GW=GW,gsurvey=gsurvey)
    return numpy.linalg.inv(numpy.array([[f00,f10,f02],[f10,f11,f12],[f02,f12,f22+sigOM0sqinv]]))[0,0]

for sigm_GW in sigm_GWs:
    if mpi:
        if (do_duration_matrix): v_ = Parallel(n_jobs=njobs)(delayed(fisher)(zmax,duration,sigm_GW,restrate_BNS,sigOM0sqinv,GW=True,gsurvey=gsurvey) for duration in durations)
        if (do_zmax_matrix): 
            duration=5.
            v_ = Parallel(n_jobs=njobs)(delayed(fisher)(zmax,duration,sigm_GW,restrate_BNS,sigOM0sqinv, GW=True,gsurvey=gsurvey) for zmax in zmaxs)
    else:
        v_ = []
        for duration in durations:
            f00,f11,f10, f02,f12,f22,lw0,lwa,bw0,bwa,Ow0,Owa,w0w0,w0wa,wawa = zintegral_fast(zmax,None,duration,sigm_GW,restrate_BNS,GW=True,gsurvey=gsurvey)
            v_.append(numpy.linalg.inv(numpy.array([[f00,f10,f02],[f10,f11,f12],[f02,f12,f22+sigOM0sqinv]]))[0,0])

    var.append(numpy.array(v_)*2*3.14/skyfrac)
    
if (do_duration_matrix): numpy.savetxt('fisher_duration_gsurvey.csv',var,delimiter=",")
if (do_zmax_matrix): numpy.savetxt('fisher_zmax_gsurvey.csv',var,delimiter=",")

# Sample variance limit
#AP I think parameters here need to be changed....but do we need them at all for the plots we want?
#zmax_var = 0.2
#duration_var=10
#sigm_var = 0.12
#f00,f11,f10, f02,f12,f22,lw0,lwa,bw0,bwa,Ow0,Owa,w0w0,w0wa,wawa = zintegral_fast(zmax_var,None,duration_var,sigm_var,restrate_BNS)
#limit = numpy.linalg.inv(numpy.array([[f00,f10,f02],[f10,f11,f12],[f02,f12,f22+sigOM0sqinv]]))[0,0]
#limit = limit*2*3.14/skyfrac
#print(numpy.sqrt(limit))

#plots
plt.rcParams["figure.figsize"] = (8,6)
Z = numpy.sqrt(var)
levels = numpy.arange(0.0,.1001,0.01)
fig, ax = plt.subplots()
cs = ax.imshow(Z,norm=colors.LogNorm(),origin='lower',interpolation='bicubic',aspect='auto',
               extent=[X.min(),X.max(), Y.min(), Y.max()],vmin=0.025,vmax=0.11)#, cmap=cm.PuBu_r)

cbar = fig.colorbar(cs)
cbar.ax.set_ylabel(r'$\sigma_\gamma$', fontsize=18)
cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
cbar.ax.yaxis.set_minor_formatter(ScalarFormatter())
cbar.ax.tick_params(labelsize=14,which='both')

cset2 = ax.contour( Z, levels, colors='k',extent=[X.min(), X.max(), Y.min(), Y.max()])
ax.clabel(cset2, inline=True, fontsize=12)

ax.set_xlabel(r'$\log{N}$', fontsize=18)
ax.set_ylabel(r'$\sigma_M$ (mag)', fontsize=18)
ax.tick_params(labelsize=14)


# cset2 = ax.contour( Z, [numpy.sqrt(LSST_08),numpy.sqrt(LSST_12)], colors='red',extent=[X.min(), X.max(), Y.min(), Y.max()])
# ax.text(4.5,0.12, 'LSST optimal', fontdict=dict(color = color ,weight = 'bold'), va="bottom", ha="left")
# ax.text(4.4+.3,0.25, 'LSST ', fontdict=dict(color = color ,weight = 'bold'), va="bottom", ha="left")


#name = 'LSST + North'
#print (numpy.sqrt(LSST_08),numpy.sqrt(LSST_12))
#cset2 = ax.contour( Z, [numpy.sqrt(LSST_08),numpy.sqrt(LSST_12)], colors='red',extent=[X.min(), X.max(), Y.min(), Y.max()])
#ax.text(4.5,0.15, name, fontdict=dict(color = color ,weight = 'bold'), va="bottom", ha="left")

#cset2 = ax.contour( Z, [limit], colors='red',extent=[X.min(), X.max(), Y.min(), Y.max()])

#print (numpy.sqrt(limit))
plt.tight_layout()
plt.savefig("surface_GW.pdf")


