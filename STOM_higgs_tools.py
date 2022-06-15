from cmath import e
from cv2 import detail_BestOf2NearestRangeMatcher
from scipy.stats.distributions import chi2
import scipy.integrate as sp_int
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import Min
np.random.seed(1)

N_b = 10e5  # Number of background events, used in generation and in fit.
b_tau = 30.  # Spoiler.


def generate_data(n_signals=400):
    ''' 
    Generate a set of values for signal and background. Input arguement sets 
    the number of signal events, and can be varied (default to higgs-like at 
    announcement). 

    The background amplitude is fixed to 9e5 events, and is modelled as an exponential, 
    hard coded width. The signal is modelled as a gaussian on top (again, hard 
    coded width and mu).
    '''
    vals = []
    vals += generate_signal(n_signals, 125., 1.5)
    vals += generate_background(N_b, b_tau)
    return vals

def generate_signal(N, mu, sig):
    ''' 
    Generate N values according to a gaussian distribution.
    '''
    return np.random.normal(loc=mu, scale=sig, size=N).tolist()

def generate_background(N, tau):
    ''' 
    Generate N values according to an exp distribution.
    '''
    return np.random.exponential(scale=tau, size=int(N)).tolist()

def get_B_chi(vals, mass_range, nbins, A, lamb):
    ''' 
    Calculates the REDUCED chi-square value of the no-signal hypothesis (i.e background
    only) for the passed values. Need an expectation - use the analyic form, 
    using the hard coded scale of the exp. That depends on the binning, so pass 
    in as argument. The mass range must also be set - otherwise, its ignored.
    '''
    bin_heights, bin_edges = np.histogram(vals, range=mass_range, bins=nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_B_expectation(bin_edges + half_bin_width, A, lamb)
    chi = 0

    # Loop over bins - all of them for now.
    for i in range(len(bin_heights)):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator

    return chi/float(nbins-2)  # B has 2 parameters.

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

## ================================================================
#    NEW, MODIFIED FUNCTIONS FOR THE QUESTION

def get_SB_chi(vals, mass_range, nbins, A, lamb, sig_mu, sig_s, sig_A):
    bin_heights, bin_edges = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_SB_expectation(bin_edges + half_bin_width, A, lamb, sig_mu, sig_s, sig_A)
    chi = 0

    # Loop over bins - all of them for now. 
    for i in range( len(bin_heights) ):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator
    
    return chi/float(nbins-3) # B has 3 parameters... Why?
    # If we assume that the estimated parameters are only A, lambda, and signal A: 3 parameters.


## ================================================================

# END OF HIGGS_TOOLS LIBRARY
#===========================================================================================#
# All in the same Python file to prevent copious import errors.
## START OF GROUP ASSIGNMENT ##
# Note to marker:
# It is suggested that you enable Word-Wrap in your IDE. In VSCode, this is Alt-Z.

# Changes to visual style of graphs & fonts
graphDirectory = "/Graphs/"
titleFont = {'fontname': 'Bodoni 72', 'size': 13}
axesFont = {'fontname': 'CMU Sans Serif', 'size': 9}
ticksFont = {'fontname': 'DM Mono', 'size': 7}
errorStyle = {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle = {'mew': 1, 'ms': 3, 'color': 'blue'}
lineStyle = {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1}
histStyle = {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}

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
'''
# %% Exercise 1A
'''
a. Generate a dataset and plot a histogram of the rest mass values, using the same binning
and range as shown in figure 1 (i.e. [104, 155] GeV with 30 bins). 
'''

vals = generate_data()  # Generate data values from given SToM code
numBins, histRange = 30, [104, 155]  # Define histogram values

vals_bin_heights, vals_bin_edges, vals_patches = plt.hist(
    vals, range=histRange, bins=numBins, **histStyle)       # Histogram generation from given hint
plt.xlabel("Rest Mass (GeV)", **axesFont)                   
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 1A: Histogram of Rest Mass Values \n with " + str(numBins) +
          " bins, for data in [" + str(histRange[0]) + "," + str(histRange[1]) + "] GeV", **titleFont)
#plt.savefig(r'Graphs/Exercise 1A 30 bins.jpg', dpi=1000)
plt.show()

# %% Exercise 1B
''' 
b. Include the statistical uncertainties. How does it compare with Fig. 1? What happens if the binning is
changed - does the signal significance appear to change by-eye?
'''
# N.B. change the number of bins to check for signal significance changes.
widthBin = (np.diff(histRange)/numBins)[0]      # Find the width of each bin
meanBinValue = [(vals_bin_edges[i] + vals_bin_edges[i+1]) /
                2 for i in range(numBins)]      # Find the mean value of each bin
vals_bin_heights, vals_bin_edges, vals_patches = plt.hist(
    vals, range=histRange, bins=numBins, **histStyle)
plt.plot(meanBinValue, vals_bin_heights, 'x', **pointStyle)

# Calculate error in histogram data, yerr is calculated by taking the square root, xerr=widthBin/2
# Source: https://root-forum.cern.ch/t/about-bin-errors-in-a-histogram-error-in-x-or-in-y/3784
plt.errorbar(meanBinValue, vals_bin_heights, xerr=widthBin/2,
             yerr=np.sqrt(abs(vals_bin_heights)), **errorStyle)
plt.xlabel("Rest Mass (GeV)", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 1B: Histogram of Uncertainties of Rest Mass Values \n with " + str(numBins) +
          " bins, for data in [" + str(histRange[0]) + "," + str(histRange[1]) + "] GeV", **titleFont)
#plt.savefig(r'Graphs/Exercise 1B Histogram.jpg', dpi=1000)
plt.show()

# %% Exercise 2A
''' 
2. Background Parameterisation
To study the Higgs signal, the background distribution must first be parameterised by an
exponential distribution: 𝐵(𝑥) = 𝐴𝑒^(t/λ). The 𝐴 parameter is a normalisation factor required
to scale the background PDF to the histogram data, whereas λ sets the gradient of the
exponential decay. For this part of the question, in order to avoid influence from the signal,
decide on an upper limit such that only background data is considered (e.g. only use values
below 120 MeV). Later, we will extrapolate this parameterisation into the higher mass region
to account for the background beneath the signal.

A few methods can be used to estimate the two background parameters:

a. It was demonstrated in lectures how to analytically estimate λ given a set of data points.
Repeat this exercise for this dataset. Does the upper cut applied to remove the signal
affect your result? Is there a way to avoid this?
'''

reduced_vals = [j for j in vals if j <= 120]    # Choosing cut-off of 120, remove any larger data

# Use Maximum Likelihood Estimator:
# Source: https://math.stackexchange.com/questions/101481/calculating-maximum-likelihood-estimation-of-the-exponential-distribution-and-pr
MLE_lambda = sum(reduced_vals)/len(reduced_vals)
print("Exercise 2A: In our exponential distribution considering below 120 only,\n λ = ", MLE_lambda)

# Choosing a new upper cut-off of 130, removing the section in the middle with the Higgs "bump"
reduced_vals = [j for j in vals if j <= 120 or j >= 130]
MLE_lambda_new = np.sum(reduced_vals)/len(reduced_vals)
print("In our exponential distribution also considering above 130,\n λ = ", MLE_lambda_new)


# Using a singular cut-off of 120 ignores signficant amounts of data, but using two cut-offs we can more accurately find a value for the parameter A. Therefore our second value will be different and more accurate.
# %% Exercise 2B
''' 
b. Given a value of λ, find A by scaling the PDF to the data such that the area beneath
the scaled PDF has equal area to the data. 
'''
# We only want to deal with the range where there isn't the 'Higgs bump'.
# Define an exponential function of form, e^(-x/lambda), with scale factor A ()= 1, by default)
def exponential(x, lambduh=MLE_lambda_new, A_scaling=1):
    return A_scaling*np.exp(-x/lambduh)

### Therefore we're looking at the range 104-120 and 130-155. We consider each separately.
intendedRangeA = [104, 120] # Looking at the range 104-120
# Non-integer number of bins in intendedRange, therefore we find closest integer value.
numBinsExpA = int(np.diff(intendedRangeA)/widthBin)
dataRangeA = [intendedRangeA[0], intendedRangeA[0]+numBinsExpA*widthBin]  # Find the range of data those bins cover

exp_bin_heights, exp_bin_edges, exp_patches = plt.hist(
    vals, range=dataRangeA, bins=numBins, **histStyle)
AreaHistA = np.sum(exp_bin_heights*widthBin) # Calculate area estimate using the binned histogram

AreaExpA = sp_int.quad(exponential, dataRangeA[0], dataRangeA[1], args=(
    MLE_lambda_new))  # Determine the area under this graph (PDF)


### Looking at the range 130-155, same method as above for [104,120]
intendedRangeB = [130, 155]
numBinsExpB = int(np.diff(intendedRangeB)/widthBin)
dataRangeB = [intendedRangeB[0], intendedRangeB[0]+numBinsExpB*widthBin]

exp_bin_heights, exp_bin_edges, exp_patches = plt.hist(
    vals, range=dataRangeB, bins=numBins, **histStyle)
AreaHistB = np.sum(exp_bin_heights*widthBin)

AreaExpB = sp_int.quad(exponential, dataRangeB[0], dataRangeB[1], args=(
    MLE_lambda_new))

A = (AreaHistA + AreaHistB)/(AreaExpA[0] + AreaExpB[0]) # Combined Histogram/Exponential Area Ratio
print("A, Scaling Factor from Area Ratio: ", A)

plt.clf()  # Clear PLT
plt.cla()

# %% Exercise 2C
'''
c. Overlay the background expectation curve onto your histogram (extrapolated over the
full mass range). How does it compare qualitatively with figure 1?
'''
# Plot Histogram Points
meanBinValue = [(vals_bin_edges[i] + vals_bin_edges[i+1]) /
                2 for i in range(numBins)]
plt.plot(meanBinValue, vals_bin_heights, 'x', **pointStyle)

# Plot the Exponential Curve, with estimated value of A (by ratios)
x = np.linspace(histRange[0], histRange[1], 10000)
plt.plot(x, exponential(x, MLE_lambda_new, A))
plt.xlabel("Rest Mass (GeV)", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 2C: Histogram with Exponential (Lambda from Areas) of Rest Mass Values \n with " +
          str(numBins) + " bins, for data in [" + str(histRange[0]) + "," + str(histRange[1]) + "] GeV", **titleFont)
plt.savefig(r'Graphs/Exercise 2C.jpg', dpi=1000)
plt.show()

# %%
'''
d. (*) Generally, parameters can be estimated from data by performing a fit. This is often
required for distributions that are difficult to parameterise analytically. Multiple trial values
of A and λ can be tested until the best agreement (e.g. measured by a binned χ2 value)
is found. Write a small program that scans a range of both and A values (i.e. a 2D search)
and track the value of the χ2 for each trial. Does the minimum χ2 correspond to the
values found using the previous estimator methods? [See also hint 6]
'''
# These values are adjustable, at the discretion of the reader.
# Choosing appropriate linspace values is left as an exercise for the reader.
# Test the local values near our estimated value for A
A_test_values = np.linspace(A-2000, A+2000, 100)
lamb_test_values = np.linspace(MLE_lambda_new-3, MLE_lambda_new+3, 100)
min_chi, min_chi_values = 1e6, [0, 0]

# This section was commented out once parameters were established
# [A = 58764.012697989645, λ = 29.886872390455476]
''' for k in range (len(A_test_values)):
    for l in range (len(lamb_test_values)):
        current_chi = get_B_chi(vals,histRange,numBins,A_test_values[k],lamb_test_values[l])
        if current_chi < min_chi:
            min_chi = current_chi
            min_chi_values = [A_test_values[k],lamb_test_values[l]] '''

# Hardcoded out the Chi squared minimisation test. Run again when desired.
min_chi_values = [58764.012697989645, 29.886872390455476] # Values obtained
newA, newLambda = min_chi_values[0],min_chi_values[1]

# Plot the original histogram points
meanBinValue = [(vals_bin_edges[i] + vals_bin_edges[i+1]) /
                2 for i in range(numBins)]
plt.plot(meanBinValue, vals_bin_heights, 'x', **pointStyle)
# Add the new fitted exponential line of best fit
x = np.linspace(histRange[0], histRange[1], 10000)
plt.plot(x, exponential(x, min_chi_values[1], min_chi_values[0]))
plt.xlabel("Rest Mass (GeV)", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 2D: Histogram with Reduced Chi^2 Exponential of Rest Mass Values \n with " +
          str(numBins) + " bins, for data in [" + str(histRange[0]) + "," + str(histRange[1]) + "] GeV", **titleFont)
#plt.savefig(r'Graphs/Exercise 2D.jpg', dpi=1000)
plt.show()

# Get the value of (minimised) Chi-Squared for these estimated parameters (multiply by degrees of freedom because get_B_chi gives reduced χ².)
minimised_chi = 28*get_B_chi(vals,histRange,numBins,newA,newLambda)
estimated_chi = 28*get_B_chi(vals,histRange,numBins,A,MLE_lambda_new)
print("From our chi-squared minimisation: χ² = ",minimised_chi)
print("From our previous estimations: χ² = ",estimated_chi)
# Make some comments here on minimum χ^2 value

# %% Exercise 3
'''
3. Goodness of Fit
a. By finding the reduced χ2 value, examine the goodness of fit of your estimated
parameters in the background only mass region.
Hint 6: The function get_B_chi(vals, (histo_range _low, histo_range_up), histo_n_bins, A,
lamb) will return the reduced χ2 value for the set of measurements vals and input background
model. Nb. since this will depend on the histogram binning, those settings also have to be
passed in the function call.
'''
# Restrict region to [104,120], with the same bin width and proceed
goodnessOfFitBins,goodnessOfFitRange = numBinsExpA, dataRangeA
goodnessOfFit = get_B_chi(vals, [104,120], goodnessOfFitBins,
                          newA, newLambda)

print('Reduced χ² for',goodnessOfFitBins,'bins in the BG region',goodnessOfFitRange,':', goodnessOfFit)

# %% Exercise 4A
'''
4. Hypothesis Testing
a. What happens if you include the signal region in a reduced χ^2 calculation for a
background-only hypothesis using the mass range shown in figure 1? What is the
corresponding alpha value (also known as p-value)? Can we exclude this hypothesis?
'''

# We repeat the same calculation as in 2C but without the restriction on our domain. We look at all values from 104-155.
redChi2Score = get_B_chi(vals, histRange, numBins,
                         newA, newLambda)
print("Reduced value of χ², including ",histRange,':', redChi2Score)

# Number of parameters is 2 (A,λ), therefore ddof is 28
# Reduced Chi-Squared Test is Chi-Square Test per Degrees of Freedom
# https://stackoverflow.com/questions/11725115/p-value-from-chi-sq-test-statistic-in-python
ddof = numBins - 2
AlphaValue = chi2.sf(redChi2Score*ddof, ddof)
print("Corresponding Alpha Value, α: ", AlphaValue)

# Make some comment about insanely small
# Make some comments here about "Can we exclude this hypothesis"
# %%
'''
b. The χ² value will vary between repeats of the simulation due to random fluctuations. In
order to understand this variation for a background-only hypothesis, repeat the
simulation many times (e.g. 10k) with the signal amplitude set to zero, and form a
distribution of the χ² values for a background only scenario. Does the distribution look
how you expect for this number of degrees of freedom? Are there any values near the
value found in question 4a, and if so, how do we interpret these values?
'''
chi2BG,numTrials = [], 10000

# About an hour's worth of runtime. Caution!
for m in range(numTrials):
    # Set signal amplitude = 0, therefore only consider background data
    DataBG = generate_background(N_b, b_tau)
    max_BG, min_BG = np.max(DataBG), np.min(DataBG)
    numBinsBG = int((max_BG - min_BG)/widthBin)
    BG_bin_heights, BG_bin_edges = np.histogram(DataBG, bins=numBinsBG)
    widthBinBG = np.abs(BG_bin_edges[1]-BG_bin_edges[0])
    plt.show()

    # Use MLE Estimation
    lambda_BG = np.sum(DataBG)/len(DataBG)
    Area_BG = np.sum(BG_bin_heights*widthBinBG)
    # Why? Using the expectation value of an exponential distribution, we find the ratio of the areas under the curve, and therefore A.
    A_BG = Area_BG/lambda_BG

    # Checking the fit of the estimated parameters against BG histogram data.
    BG_bin_edges_new = np.delete(BG_bin_edges,-1)
    BG_bin_edges_new = BG_bin_edges_new + 0.5*widthBinBG
    
    #x = np.linspace(min_BG,max_BG,1000)
    #plt.plot(BG_bin_edges_new,BG_bin_heights,'x',**pointStyle)
    #plt.plot(x,exponential(x,lambda_BG,A_BG))
    #plt.show()

    redChi2BG = get_B_chi(DataBG, [min_BG,max_BG], numBinsBG, A_BG, lambda_BG)
    chi2BG.append(redChi2BG)

plt.hist(chi2BG,bins = 30,**histStyle)
plt.xlabel("Reduced Chi Squared Value", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 4B: Distributions of Chi^2 Values", **titleFont)
plt.savefig(r'Graphs/Exercise 4B.jpg', dpi=1000)
plt.show()

# %%
'''
c. (*) Repeat part b for a variety of signal amplitudes. What signal amplitude is required
such that its expected p value equals 0.05? Supposing we call any p-value < 0.05 a hint
of a signal; what are the chances of finding a hint if the expected p-value equals 0.05?
'''

# To do, run overnight. Check against the generated background data, added to the Gaussian, assuming that the Gaussian centered at 125GeV, and with a standard deviation of 1.5GeV.

chi2SB,numTrials = [], 1     # Run fewer trials for each value of amplitude.

# About an hour's worth of runtime. Caution!
for m in range(numTrials):
    # Set signal amplitude = 0, therefore only consider background data
    DataSB = generate_data(400)
    max_SB, min_SB = np.max(DataSB), np.min(DataSB)
    numBinsSB = int((max_SB - min_SB)/widthBin)
    SB_bin_heights, SB_bin_edges = np.histogram(DataSB, bins=numBinsSB)
    widthBinSB = np.abs(SB_bin_edges[1]-SB_bin_edges[0])
    plt.show()

    # Use MLE Estimation
    lambda_SB = np.sum(DataSB)/len(DataSB)
    Area_SB = np.sum(SB_bin_heights*widthBinSB)
    # Why? Using the expectation value of an exponential distribution, we find the ratio of the areas under the curve, and therefore A.
    A_SB = Area_SB/lambda_SB

    # Checking the fit of the estimated parameters against SB histogram data.
    SB_bin_edges_new = np.delete(SB_bin_edges,-1)
    SB_bin_edges_new = SB_bin_edges_new + 0.5*widthBinSB
    
    #x = np.linspace(min_SB,max_SB,1000)
    #plt.plot(SB_bin_edges_new,SB_bin_heights,'x',**pointStyle)
    #plt.plot(x,exponential(x,lambda_SB,A_SB))
    #plt.show()

    redChi2SB = get_B_chi(DataSB, [min_SB,max_SB], numBinsSB, A_SB, lambda_SB)
    chi2SB.append(redChi2SB)

plt.hist(chi2BG,bins = 30,**histStyle)
plt.xlabel("Reduced Chi Squared Value", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 4B: Distributions of Chi^2 Values", **titleFont)
plt.savefig(r'Graphs/Exercise 4B.jpg', dpi=1000)
plt.show()


# %%
'''
5. Signal Estimation
Given the background parameterisation, we can test the consistency of a background plus
signal hypothesis. The signal expectation can often be provided by other simulations, as with
the example in the first part of this question.

a. Recalculate the c2 value for a background + signal hypothesis, presuming the signal
can be characterised by the following parameters: gaus(A_signal = 700, μ = 125 GeV,
s = 1.5 GeV). Comment on the ‘goodness of fit’ and alpha value?
'''
signals = np.linspace(histRange[0],histRange[1],10000)
A_signal, mu_signal, sig_signal = 700, 125, 1.5

plt.plot(meanBinValue, vals_bin_heights, 'x', **pointStyle)
plt.plot(signals, get_SB_expectation(signals,min_chi_values[0],min_chi_values[1],mu_signal,sig_signal,A_signal))
plt.errorbar(meanBinValue, vals_bin_heights, xerr=widthBin/2,
             yerr=np.sqrt(abs(vals_bin_heights)), **errorStyle)
plt.xlabel("Rest Mass (GeV)", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 5A: Exponential-Gaussian Fit \n with " +
          str(numBins) + " bins, for data in [" + str(histRange[0]) + "," + str(histRange[1]) + "] GeV", **titleFont)
#plt.savefig(r'Graphs/Exercise 5A.jpg', dpi=1000)
plt.show()

redChi2SB = get_SB_chi(vals, histRange, numBins, min_chi_values[0], min_chi_values[1], mu_signal, sig_signal, A_signal)
print("Goodness of Fit for our Values of χ² on Exponential-Gaussian: ",redChi2SB)


# Number of parameters is 5 (A,λ), therefore ddof is 25
# Reduced Chi-Squared Test is Chi-Square Test per Degrees of Freedom
ddof = numBins - 5
AlphaValue = chi2.sf(redChi2SB*ddof, ddof)
print("Corresponding Alpha Value, α: ", AlphaValue)

# %%
'''
b. (*) Using similar techniques to the background parameterisation, justify these signal
parameters.
'''

def get_B_chi_signal_1(vals, mass_range, nbins, A, lamb, mu, sig, signal_amp):
    ''' 
    Calculates the chi-square value of the no-signal hypothesis (i.e background
    only) for the passed values. Need an expectation - use the analyic form, 
    using the hard coded scale of the exp. That depends on the binning, so pass 
    in as argument. The mass range must also be set - otherwise, its ignored.
    '''
    bin_heights, bin_edges = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_SB_expectation(bin_edges + half_bin_width, A, lamb, mu, sig, signal_amp)
    chi = 0

    # Loop over bins - all of them for now. 
    for i in range( len(bin_heights) ):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator
    
    nzeros=0
    for i in range(0,nbins):
        if bin_heights[i]==0:
            nzeros+=1
        else:
            continue
    ndata=len(bin_heights)-nzeros
    
    return chi/float(ndata-1) # B has 1 parameters.

    

A_sig_test_values = np.linspace(A_signal-50, A_signal+50, 100)
mu_sig_test_values = np.linspace(mu_signal-5, mu_signal+5, 100)
s_sig_test_values = np.linspace(sig_signal-1, sig_signal+1, 10)
min_chi_SB, min_chi_values_SB = 1e6, [0, 0, 0]
a = 0

# This section was commented out once parameters were established
# [A = 58764.012697989645, λ = 29.886872390455476]
for k in range (len(A_sig_test_values)):
    for l in range (len(mu_sig_test_values)):
        for m in range(len(s_sig_test_values)):
            current_chi_SB = get_SB_chi(vals,histRange,numBins,min_chi_values[0],min_chi_values[1],A_sig_test_values[k],mu_sig_test_values[l],s_sig_test_values[m])

            a += 1
            print(a,'/',100*100*10)
            
            if current_chi_SB < min_chi_SB:
                min_chi_SB = current_chi_SB
                min_chi_values_SB = [A_sig_test_values[k],mu_sig_test_values[l],s_sig_test_values[m]]
                
                

print(min_chi_values_SB)

# %%
'''
c. (**) We tested the background + signal hypothesis for a known signal mass position. In
many searches, the mass isn’t known in advance. Write a program to loop over a range
of masses and plot the corresponding value of the c 2 of a signal + background
hypothesis (keeping the signal amplitude and gaussian width parameters fixed). How
does the significance of the observed signal at 125 MeV change given we scanned the
whole mass range?
'''


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
