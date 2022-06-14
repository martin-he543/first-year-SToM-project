import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

N_b = 10e5 # Number of background events, used in generation and in fit.
b_tau = 30. # Spoiler.

def generate_data(n_signals = 400):
    ''' 
    Generate a set of values for signal and background. Input arguement sets 
    the number of signal events, and can be varied (default to higgs-like at 
    announcement). 
    
    The background amplitude is fixed to 9e5 events, and is modelled as an exponential, 
    hard coded width. The signal is modelled as a gaussian on top (again, hard 
    coded width and mu).
    '''
    vals = []
    vals += generate_signal( n_signals, 125., 1.5)
    vals += generate_background( N_b, b_tau)
    return vals

def generate_signal(N, mu, sig):
    ''' 
    Generate N values according to a gaussian distribution.
    '''
    return np.random.normal(loc = mu, scale = sig, size = N).tolist()

def generate_background(N, tau):
    ''' 
    Generate N values according to an exp distribution.
    '''
    return np.random.exponential(scale = tau, size = int(N)).tolist()

def get_B_chi(vals, mass_range, nbins, A, lamb):
    ''' 
    Calculates the chi-square value of the no-signal hypothesis (i.e background
    only) for the passed values. Need an expectation - use the analyic form, 
    using the hard coded scale of the exp. That depends on the binning, so pass 
    in as argument. The mass range must also be set - otherwise, its ignored.
    '''
    bin_heights, bin_edges = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_B_expectation(bin_edges + half_bin_width, A, lamb)
    chi = 0

    # Loop over bins - all of them for now. 
    for i in range( len(bin_heights) ):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator
    
    return chi/float(nbins-2) # B has 2 parameters.

def get_B_expectation(xs, A, lamb):
    ''' 
    Return a set of expectation values for the background distribution for the 
    passed in x values. 
    '''
    return [A*np.exp(-x/lamb) for x in xs]

def signal_gaus(x, mu, sig, signal_amp):
    return signal_amp/(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def get_SB_expectation(xs, A, lamb, mu, sig, signal_amp):
    ys = []
    for x in xs:
        ys.append(A*np.exp(-x/lamb) + signal_gaus(x, mu, sig, signal_amp))
    return ys


### END OF HIGGS_TOOLS LIBRARY
#===========================================================================================#
# All in the same Python file to prevent copious import errors.
## START OF GROUP ASSIGNMENT ##

graphDirectory = "/Graphs/"
# Adjust your font settings
titleFont = {'fontname':'Bodoni 72','size':13}
axesFont = {'fontname':'CMU Sans Serif','size':9}
ticksFont = {'fontname':'DM Mono','size':7}
errorStyle = {'mew':1,'ms':3,'capsize':3,'color':'blue','ls':''}
pointStyle = {'mew':1,'ms':3,'color':'blue'}
lineStyle = {'linewidth':0.5}
histStyle = {'facecolor':'blue','alpha':0.5,'edgecolor':'black'}

''' 
1. Generating Simulated Data

Download the python script named STOM_higgs_tools.py. The file is well documented - read
through this to become familiar with the examples and functions available.
The function named generate_data(n_signals = 400) will create a simulated data-set, and
return a python list of positive rest mass values (with no other range restrictions applied). The
signal amplitude can be varied by the argument. n_signals - the default of 400 events
corresponds to the real-life case around the time of the discovery, and the number of
background entries is fixed. The dataset will change due to random statistical fluctuations if
the function is called again, but the data will not change between executions of the problem
(i.e. the random seed is fixed at the start of the tools module).

a. Generate a dataset and plot a histogram of the rest mass values, using the same binning
and range as shown in figure 1 (i.e. [104, 155] GeV with 30 bins). 
'''
#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate as sp_int

# Each list entry represents the rest mass reconstructed from a collision.
vals = generate_data()
numBins,histRange = 30, [104,155]

vals_bin_heights,vals_bin_edges,vals_patches = plt.hist(vals,range=histRange,bins=numBins,**histStyle)
plt.xlabel("Rest Mass (GeV)", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 1A: Histogram of Rest Mass Values \n with " + str(numBins) + " bins, for data in [" + str(histRange[0]) + "," + str(histRange[1]) + "] GeV", **titleFont)
#plt.savefig(graphDirectory + 'Exercise 1A.png', dpi=500)
plt.show()

''' 
b. Include the statistical uncertainties. How does it compare with Fig. 1? What happens if the binning is
changed - does the signal significance appear to change by-eye?
'''
#%%
# N.B. change the number of bins to check for signal significance changes.
# Find the width of each bin
widthBin = (np.diff(histRange)/numBins)[0]
# Find the mean value of each bin
meanBinValue = [(vals_bin_edges[i] + vals_bin_edges[i+1])/2 for i in range (numBins)]
plt.plot(meanBinValue,vals_bin_heights,'x',**pointStyle)
# Calculate error in histogram data, yerr is calculated by taking the square root, xerr=widthBin/2
# Source: https://root-forum.cern.ch/t/about-bin-errors-in-a-histogram-error-in-x-or-in-y/3784
plt.errorbar(meanBinValue,vals_bin_heights, yerr=np.sqrt(abs(vals_bin_heights)),**errorStyle)
plt.xlabel("Rest Mass (GeV)", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 1B: Histogram of Uncertainties of Rest Mass Values \n with " + str(numBins) + " bins, for data in [" + str(histRange[0]) + "," + str(histRange[1]) + "] GeV", **titleFont)
#plt.savefig(graphDirectory + 'Exercise 1B.png', dpi=500)
plt.show()

''' 
2. Background Parameterisation
To study the Higgs signal, the background distribution must first be parameterised by an
exponential distribution: 𝐵(𝑥) = 𝐴𝑒 !"/$ . The 𝐴 parameter is a normalisation factor required
to scale the background PDF to the histogram data, whereas l sets the gradient of the
exponential decay. For this part of the question, in order to avoid influence from the signal,
decide on an upper limit such that only background data is considered (e.g. only use values
below 120 MeV). Later, we will extrapolate this parameterisation into the higher mass region
to account for the background beneath the signal.

A few methods can be used to estimate the two background parameters:

a. It was demonstrated in lectures how to analytically estimate l given a set of data points.
Repeat this exercise for this dataset. Does the upper cut applied to remove the signal
affect your result? Is there a way to avoid this?
'''
#%%
# Choosing cut-off of 120, remove any larger data
reduced_vals = [j for j in vals if j<= 120]
# Use Maximum Likelihood Estimator:
# Source: https://math.stackexchange.com/questions/101481/calculating-maximum-likelihood-estimation-of-the-exponential-distribution-and-pr
MLE_lambda = sum(reduced_vals)/len(reduced_vals)
print("Exercise 2A: In our exponential distribution considering below 120 only,\n λ = ",MLE_lambda)

# Choosing upper cut-off of 130, removing the section in the middle with the Higgs "bump"
reduced_vals = [j for j in vals if j<= 120 or j >= 130]
MLE_lambda_new = np.sum(reduced_vals)/len(reduced_vals)
print("In our exponential distribution also considering above 130,\n λ = ",MLE_lambda_new)


''' 
b. Given a value of λ, find A by scaling the PDF to the data such that the area beneath
the scaled PDF has equal area to the data. 
'''
#%%
# Indicate the range of data we will take for estimation
intendedRange = [0,120]
# As not a whole number of bins fit in intendedRange, this is the actual range we will use.
numBinsExp = int(np.diff(intendedRange)/widthBin)
dataRange = [0,numBinsExp*widthBin]

# Calculate an estimation of area using the binned histogram
exp_bin_heights,exp_bin_edges,exp_patches = plt.hist(vals,range=dataRange,bins=numBins,**histStyle)
AreaHist = np.sum(exp_bin_heights*widthBin)

# Define an exponential function of form, e^(-x/lambda)
# Use our previous MLE estimation, as well as A = 1 for comparison
def exponential (x, lambduh = MLE_lambda_new, A_scaling=1):
    return A_scaling*np.exp(-x/lambduh)
# Determine the area under this graph (i.e. the PDF of an exponential), in dataRange
AreaExp = sp_int.quad(exponential,dataRange[0],dataRange[1], args=(MLE_lambda_new))

A = AreaHist/AreaExp[0]
print("Scaling Factor: ", A)

'''
c. Overlay the background expectation curve onto your histogram (extrapolated over the
full mass range). How does it compare qualitatively with figure 1?
'''
# %%
x = np.linspace(histRange[0],histRange[1],10000)
widthBin = (np.diff(histRange)/numBins)[0]
# Find the mean value of each bin
meanBinValue = [(vals_bin_edges[i] + vals_bin_edges[i+1])/2 for i in range (numBins)]
plt.plot(meanBinValue,vals_bin_heights,'x',**pointStyle)
plt.plot(x,exponential(x,MLE_lambda_new,A),**lineStyle)
plt.xlabel("Rest Mass (GeV)", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 1A: Histogram of Rest Mass Values \n with " + str(numBins) + " bins, for data in [" + str(histRange[0]) + "," + str(histRange[1]) + "] GeV", **titleFont)
#plt.savefig(graphDirectory + 'Exercise 1A.png', dpi=500)
plt.show()


'''
d. (*) Generally, parameters can be estimated from data by performing a fit. This is often
required for distributions that are difficult to parameterise analytically. Multiple trial values
of A and l can be tested until the best agreement (e.g. measured by a binned c 2 value)
is found. Write a small program that scans a range of both and A values (i.e. a 2D search)
and track the value of the c2 for each trial. Does the minimum c2 correspond to the
values found using the previous estimator methods? [See also hint 6]
'''
# %%


'''
3. Goodness of Fit
a. By finding the reduced c 2 value, examine the goodness of fit of your estimated
parameters in the background only mass region.
Hint 6: The function get_B_chi(vals, (histo_range _low, histo_range_up), histo_n_bins, A,
lamb) will return the reduced
c2 value for the set of measurements vals and input background
model. Nb. since this will depend on the histogram binning, those settings also have to be
passed in the function call.
'''

# %%



'''
4. Hypothesis Testing
a. What happens if you include the signal region in a reduced c 2 calculation for a
background-only hypothesis using the mass range shown in figure 1? What is the
corresponding alpha value (also known as p-value)? Can we exclude this hypothesis?
'''

#%%

'''
b. The c 2 value will vary between repeats of the simulation due to random fluctuations. In
order to understand this variation for a background-only hypothesis, repeat the
simulation many times (e.g. 10k) with the signal amplitude set to zero, and form a
distribution of the c 2 values for a background only scenario. Does the distribution look
how you expect for this number of degrees of freedom? Are there any values near the
value found in question 4a, and if so, how do we interpret these values?
'''
# %%


'''
c. (*) Repeat part b for a variety of signal amplitudes. What signal amplitude is required
such that its expected p value equals 0.05? Supposing we call any p-value < 0.05 a hint
of a signal; what are the chances of finding a hint if the expected p-value equals 0.05?
'''
#%%

'''
5. Signal Estimation
Given the background parameterisation, we can test the consistency of a background plus
signal hypothesis. The signal expectation can often be provided by other simulations, as with
the example in the first part of this question.

a. Recalculate the c2 value for a background + signal hypothesis, presuming the signal
can be characterised by the following parameters: gaus(A_signal = 700, μ = 125 GeV,
s = 1.5 GeV). Comment on the ‘goodness of fit’ and alpha value?
'''
# %%

'''
b. (*) Using similar techniques to the background parameterisation, justify these signal
parameters.
'''
# %%

'''
c. (**) We tested the background + signal hypothesis for a known signal mass position. In
many searches, the mass isn’t known in advance. Write a program to loop over a range
of masses and plot the corresponding value of the c 2 of a signal + background
hypothesis (keeping the signal amplitude and gaussian width parameters fixed). How
does the significance of the observed signal at 125 MeV change given we scanned the
whole mass range?
'''

# %%

'''
5. Conclusions
We have now tested both a background-only hypothesis, and a background + signal
hypothesis for the simulated dataset. Using your answers to the above, is there sufficient
evidence to claim the existence of a new particle (were this a real experiment of course)?
What could help further build confidence to the claim? So far, we have assumed that statistical
(random) uncertainties have dominated over systematic uncertainties - how might the
evidence change had systematic uncertainties been larger?
'''


print("Fin")