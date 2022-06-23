import numpy as np
import scipy as sp
from scipy.stats.distributions import chi2
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
np.random.seed(1)

N_b = 10e5  # Number of background events, used in generation and in fit.
b_tau = 30.  # Spoiler.

def generate_data(n_signals=400):
    ''' 
    Generate a set of values for signal and background. Input argument sets 
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

#%%
# END OF HIGGS_TOOLS LIBRARY
#=========================================================================#
# All in the same Python file to prevent copious import errors.
## START OF GROUP ASSIGNMENT ##
# Note to marker:
# It is suggested that you enable Word-Wrap in your IDE. In VSCode, this is Alt-Z.

# Custom stylings (graphs and otherwise)
graphDirectory = "/Graphs/"
titleFont = {'fontname': 'Bodoni 72', 'size': 13}
axesFont = {'fontname': 'CMU Sans Serif', 'size': 11}
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
# %% 
# Exercise 1A
'''
a. Generate a dataset and plot a histogram of the rest mass values, using the same binning
and range as shown in figure 1 (i.e. [104, 155] GeV with 30 bins). 
'''

vals = generate_data()  # Generate data values from given SToM code
numBins, histRange = 30, [104, 155]  # Define histogram values

vals_bin_heights, vals_bin_edges, vals_patches = plt.hist(
    vals, range=histRange, bins=numBins, **histStyle) # Histogram generation (per Hint 1)
plt.xlabel("Rest Mass (GeV/c²)", **axesFont) # Axes labelling
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont) # Format the axes ticks
plt.yticks(**ticksFont)
plt.title("Exercise 1A: Histogram of Rest Mass Values in 30 bins",**titleFont)
# plt.title("Exercise 1A: Histogram of Rest Mass Values \n with " + str(numBins) + " bins, for data in [" + str(histRange[0]) + "," + str(histRange[1]) + "] GeV", **titleFont)
#plt.savefig(r'Graphs/Final Selection/Exercise 1A 30 bins.jpg', dpi=1000) # Save high-res image
plt.show()

# %% 
# Exercise 1B - fully notated, again consider x-axis labelling
''' 
b. Include the statistical uncertainties. How does it compare with Fig. 1? What happens if the binning is
changed - does the signal significance appear to change by-eye? 
(Increasing number of bins while fixing number of data points to 400 improves the clarity of the gaussian signal as histogram approaches a contununous distribution).
'''
# N.B. can change "numBins" to check for signal significance changes.
widthBin = (np.diff(histRange)/numBins)[0]      # Find the width of each bin
meanBinValue = [(vals_bin_edges[i] + vals_bin_edges[i+1]) /
                2 for i in range(numBins)]      # Find the mean value of each bin

plt.errorbar(meanBinValue, vals_bin_heights, xerr=widthBin/2,
             yerr=np.sqrt(abs(vals_bin_heights)), **errorStyle) # Calculate error in histogram data, yerr is calculated by taking the square root, xerr=widthBin/2
# Source: https://root-forum.cern.ch/t/about-bin-errors-in-a-histogram-error-in-x-or-in-y/3784
plt.plot(meanBinValue, vals_bin_heights, 'x', **pointStyle) # Plot the mean values

# Plot identical histogram as in Exercise 1A
vals_bin_heights, vals_bin_edges, vals_patches = plt.hist(
    vals, range=histRange, bins=numBins, **histStyle)
plt.xlabel("Rest Mass (GeV/c²)", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 1B: Uncertainties in Rest Mass Values \n with " + str(numBins) +
          " bins, for data in [" + str(histRange[0]) + "," + str(histRange[1]) + "] GeV", **titleFont)
#plt.savefig(r'Graphs/Final Selection/Exercise 1B.jpg', dpi=1000)
plt.show()

# Indeed as the binning is changed, the statistical uncertainties decrease as the number of bins are increased, and the histogram approaches a continuous distribution. This can be seen clearly by changing the value of "numBins" in Exercise 1A. This is as expected, increasing the clarity of the Gaussian and exponential signals .

# %% 
# Exercise 2A
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

reduced_vals = [j for j in vals if j <= 120]    # Choosing cut-off of 120, remove data < 120

# Use Maximum Likelihood Estimator:
# Assuming our exponential of form Ae^(-x/λ) to be of A = k/λ, for k is a constant, then we have that the MLE of λ can be determined by solving the differential of the log-likelihood function of λ. This gives an MLE of λ = 1/N (Σ(i=1,N)t_i), where t_i is the "number of entries" (height) for each bin. This is given below:
MLE_lambda = sum(reduced_vals)/len(reduced_vals) # Finds estimate of λ
print("Exercise 2A: In our exponential distribution considering below 120 only,\n λ = ", MLE_lambda)

# Restricting our range to exclusively before 120 evidently affects our result; it discounts  any data past this threshold, and consequently results in an underestimate; therefore we should consider more data beyond 120.
# Choose a new upper cut-off of 130, removing the middle section with the Higgs "bump"
reduced_vals = [j for j in vals if j <= 120 or j >= 130] # Alternative estimate of λ
MLE_lambda_new = np.sum(reduced_vals)/len(reduced_vals)
print("In our exponential distribution also considering above 130,\n λ = ", MLE_lambda_new)

# Using a singular cut-off of 120 ignores signficant amounts of data, but using two cut-offs we can more accurately find a value for the parameter A. Therefore our second value will be different (increased) and more accurate.

# %% 
# Exercise 2B
''' 
b. Given a value of λ, find A by scaling the PDF to the data such that the area beneath
the scaled PDF has equal area to the data. 
'''
# We only want to deal with the range where there isn't the 'Higgs bump'. Therefore we're looking at the range 104-120 and 130-155. We consider each separately.
# Define an exponential function of form, e^(-x/lambda), with scale factor A = 1, by default
def exponential(x, lambduh=MLE_lambda_new, A_scaling=1):
    return A_scaling*np.exp(-x/lambduh)

intendedRangeA = [104, 120] # Looking at the range 104-120
# Non-integer number of bins in intendedRange, therefore we find floored integer value.
numBinsExpA = int(np.diff(intendedRangeA)/widthBin) # Calculate integer number of bins
dataRangeA = [intendedRangeA[0], intendedRangeA[0]+numBinsExpA*widthBin]  # Find range covered

exp_bin_heights, exp_bin_edges, exp_patches = plt.hist(
    vals, range=dataRangeA, bins=numBinsExpA, **histStyle) # Generate histogram in range
AreaHistA = np.sum(exp_bin_heights*widthBin) # Calculate area using the binned histogram
AreaExpA = sp_int.quad(exponential, dataRangeA[0], dataRangeA[1], args=(
    MLE_lambda_new))  # Determine the area (i.e. CDF) under this graph (of the PDF) 

intendedRangeB = [130, 155] # Looking at the range 130-155, proceed as before
numBinsExpB = int(np.diff(intendedRangeB)/widthBin)
dataRangeB = [intendedRangeB[0], intendedRangeB[0]+numBinsExpB*widthBin]
exp_bin_heights, exp_bin_edges, exp_patches = plt.hist(
    vals, range=dataRangeB, bins=numBinsExpB, **histStyle)
AreaHistB = np.sum(exp_bin_heights*widthBin)
AreaExpB = sp_int.quad(exponential, dataRangeB[0], dataRangeB[1], args=(
    MLE_lambda_new))

# Calculate the combined Histogram/Exponential Area Ratio
A = (AreaHistA + AreaHistB)/(AreaExpA[0] + AreaExpB[0])
print("A, Scaling Factor from Area Ratio: ", A)

plt.xlabel("Rest Mass (GeV/c²)", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 2B: Histogram Area Estimation", **titleFont)
#plt.savefig(r'Graphs/Final Selection/Exercise 2B.jpg', dpi=1000)
plt.show()

# %% 
# Exercise 2C
'''
c. Overlay the background expectation curve onto your histogram (extrapolated over the
full mass range). How does it compare qualitatively with figure 1?
'''
plt.clf()  # Clear previous histogram plots (just to make sure ;).
plt.cla()

# Plot Histogram Points
vals_bin_heights, vals_bin_edges, vals_patches = plt.hist(
    vals, range=histRange, bins=numBins, **histStyle)
plt.plot(meanBinValue, vals_bin_heights, 'x', **pointStyle)
plt.errorbar(meanBinValue, vals_bin_heights, xerr=widthBin/2,
             yerr=np.sqrt(abs(vals_bin_heights)), **errorStyle)

# Overlay the Background Exponential Curve, with estimated value of A (by ratio), and λ
x = np.linspace(histRange[0], histRange[1], 10000)
plt.plot(x, exponential(x, MLE_lambda_new, A), linewidth="2")


plt.xlabel("Rest Mass (GeV/c²)", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 2C: Rest Mass Values, with Background Expectation Curve", **titleFont)
#plt.title("Exercise 2C: Histogram of Rest Mass Values with Estimated Exponential \n with " + str(numBinsExpA + numBinsExpB) + " bins, for data in [" + str(histRange[0]) + "," + str(histRange[1]) + "] GeV", **titleFont)
#plt.savefig(r'Graphs/Final Selection/Exercise 2C.jpg', dpi=1000)
plt.show()

# %%
# Exercise 2D
'''
d. (*) Generally, parameters can be estimated from data by performing a fit. This is often
required for distributions that are difficult to parameterise analytically. Multiple trial values
of A and λ can be tested until the best agreement (e.g. measured by a binned χ2 value)
is found. Write a small program that scans a range of both lambda and A values (i.e. a 2D search)
and track the value of the χ2 for each trial. Does the minimum χ2 correspond to the
values found using the previous estimator methods? [See also hint 6]
'''
# These values are adjustable, at the discretion of the reader.
# Choosing appropriate linspace values is left as an exercise for the reader.
# Test the local values near our estimated value for A
A_test_values = np.linspace(A-2000, A+2000, 100) # Examine values of A nearby
lamb_test_values = np.linspace(MLE_lambda_new-1, MLE_lambda_new+1, 100)
min_chi, min_chi_values = 1e6, [0, 0] # Pick a large value of min_chi so always bigger

# REDO this section with restricted hist range [104,120] if time allows. 
# This section was commented out once parameters were established
# Attempt 1: [A = 58764.012697989645, λ = 29.886872390455476]  χ² =  73.5484274171061]
# Attempt 2: [A = 61490.57277268083, λ = 29.56364006722315] χ² =  72.93097953414986]
# Attempt 3 [FINAL]: see below for comment
''' for k in range (len(A_test_values)): # Cycle through each iteration of A in linspace
    for l in range (len(lamb_test_values)): 
        #current_chi = get_B_chi(vals,histRange,numBins,A_test_values[k],lamb_test_values[l])
        current_chi = get_B_chi(vals,intendedRangeA,numBinsExpA,A_test_values[k],lamb_test_values[l])
        if current_chi < min_chi: # Compare the values of Chi, to find minimum
            min_chi = current_chi
            min_chi_values = [A_test_values[k],lamb_test_values[l]] # Find values of A, λ '''

# Hardcoded out the Chi squared minimisation test. Run again when desired.
#min_chi_values = [58764.012697989645, 29.886872390455476]
#min_chi_values = [61490.57277268083, 29.56364006722315]
#min_chi_values = 59229.169131928524, 29.840496208885927
min_chi_values = [63188.76509152448 , 29.476859845249564]
newA, newLambda = min_chi_values[0],min_chi_values[1]

# Plot the original histogram points
meanBinValue = [(vals_bin_edges[i] + vals_bin_edges[i+1]) /
                2 for i in range(numBins)]
plt.plot(meanBinValue, vals_bin_heights, 'x', **pointStyle)
plt.errorbar(meanBinValue, vals_bin_heights, xerr=widthBin/2,
             yerr=np.sqrt(abs(vals_bin_heights)), **errorStyle)

# Add the new fitted exponential line of best fit
x = np.linspace(histRange[0], histRange[1], 10000)
plt.plot(x, exponential(x, newLambda, newA), **lineStyleBold)
plt.xlabel("Rest Mass (GeV/c²)", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 2D: Rest Mass Values, with Exponential from X² Minimisation Method", **titleFont)
#plt.title("Exercise 2D: Histogram with Reduced Chi^2 Exponential of Rest Mass Values \n with " + str(numBins) + " bins, for data in [" + str(histRange[0]) + "," + str(histRange[1]) + "] GeV", **titleFont)
#plt.savefig(r'Graphs/Final Selection/Exercise 2D.jpg', dpi=1000)
plt.show()

# Get the value of (minimised) Chi-Squared for these estimated parameters (multiply by degrees of freedom because get_B_chi gives reduced χ².)
# Conducted over the no-signal mass energy range with upper restriction [104,120]. 
# minimised_chi = get_B_chi(vals,histRange,numBins,newA,newLambda)
# estimated_chi = get_B_chi(vals,histRange,numBins,A,MLE_lambda_new)
minimised_chi = get_B_chi(vals,intendedRangeA,numBinsExpA,newA,newLambda)
estimated_chi = get_B_chi(vals,intendedRangeA,numBinsExpA,A,MLE_lambda_new)

print("From our chi-squared minimisation: χ² = ",minimised_chi*28)
print("for value of A:", newA, ", λ:",newLambda)
print("reduced χ² = ",minimised_chi)
print("From our previous estimations of A, λ: χ² = ",estimated_chi*28)
print("for value of A:", A, ", λ:",MLE_lambda_new)
print("reduced χ² = ",estimated_chi)

# %% 
# Exercise 3
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
# Note that this is done because with existing code, it is not possible to conduct a "get_B_chi" test over the entire range of [104,155] without including "Higgs bump"
goodnessOfFitBins,goodnessOfFitRange = numBinsExpA, dataRangeA
goodnessOfFit = get_B_chi(vals, [104,120], goodnessOfFitBins,
                          newA, newLambda)

print('Reduced χ² for',goodnessOfFitBins,'bins in the BG region',goodnessOfFitRange,':', goodnessOfFit)
ddof = numBinsExpA - 2  # Number of parameters is 2 (A,λ), therefore ddof is 28

pValue = chi2.sf(goodnessOfFit*ddof, ddof) # Print p-value for comparison
print("Corresponding Alpha Value, with ν=",ddof," p-value=", pValue)

# %% 
# Exercise 4A
'''
4. Hypothesis Testing
a. What happens if you include the signal region in a reduced χ^2 calculation for a
background-only hypothesis using the mass range shown in figure 1? What is the
corresponding alpha value (also known as p-value)? Can we exclude this hypothesis?
'''

# We repeat the same calculation as in 3a but without the restriction on our domain. We look at all values from 104-155.
redChi2Score = get_B_chi(vals, histRange, numBins,
                         newA, newLambda)
print("Reduced value of χ², including ",histRange,':', redChi2Score)

# Reduced Chi-Squared Test is Chi-Square Test per Degrees of Freedom. Sources:
# https://stackoverflow.com/questions/11725115/p-value-from-chi-sq-test-statistic-in-python
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm
ddof = numBins - 2  # Number of parameters is 2 (A,λ), therefore ddof is 28
pValue = chi2.sf(redChi2Score*ddof, ddof) # The survival function, 1-cdf, can ascertain p
print("Corresponding p-value, with ν=",ddof," p-value=", pValue)

# Given the null hypothesis that there is no signal, this means that due to the corresponding p-value, it is exceedingly unlikely that we are to find our p-value to be within the rejection region, so it is likely for us to reject our null hypothesis in favour of our alternative hypothesis, that there is indeed a signal (i.e. Higgs bump).

# %% 
# Exercise 4B
'''
b. The χ² value will vary between repeats of the simulation due to random fluctuations. In
order to understand this variation for a background-only hypothesis, repeat the
simulation many times (e.g. 10k) with the signal amplitude set to zero, and form a
distribution of the χ² values for a background only scenario. Does the distribution look
how you expect for this number of degrees of freedom? Are there any values near the
value found in question 4a, and if so, how do we interpret these values?
'''
chi2BG,numTrials = [], 1 # Change to 10,000 when you have patience(!)

for m in range(numTrials): # About an hour's worth of runtime (check numTrials). Caution!
    # Set signal amplitude = 0, therefore only consider background data
    DataBG = generate_background(N_b, b_tau) # Use built-in background generation tool
    max_BG, min_BG = np.max(DataBG), np.min(DataBG) # Define min/max points in DataBG
    numBinsBG = int((max_BG - min_BG)/widthBin) # Integer number of bins in this range
    BG_bin_heights, BG_bin_edges = np.histogram(DataBG, bins=numBinsBG)
    widthBinBG = np.abs(BG_bin_edges[1]-BG_bin_edges[0]) # Constructs histogram of data
    #plt.show()

    # Use MLE Estimation
    lambda_BG = np.sum(DataBG)/len(DataBG) # MLE estimator for λ in an exponential
    Area_BG = np.sum(BG_bin_heights*widthBinBG) # Estimating the histogram area
    A_BG = Area_BG/lambda_BG # Can ascertain A from a simple ratio of area to λ

    # Checking the fit of the estimated parameters against BG histogram data.
    BG_bin_edges_new = np.delete(BG_bin_edges,-1) # Remove last extra edge value
    BG_bin_edges_new = BG_bin_edges_new + 0.5*widthBinBG # Find the mean of each point
    # Uncomment for testing the fit of MLE estimation best-fit lines
    #x = np.linspace(min_BG,max_BG,1000)
    #plt.plot(BG_bin_edges_new,BG_bin_heights,'x',**pointStyle)
    #plt.plot(BG_bin_edges_new,BG_bin_heights,'x',**pointStyle)
    #plt.plot(x,exponential(x,lambda_BG,A_BG))
    #plt.show()

    redChi2BG = get_B_chi(DataBG, [min_BG,max_BG], numBinsBG, A_BG, lambda_BG)
    #print("reduced χ²:",redChi2BG)
    chi2BG.append(redChi2BG) # For each set of data, calculate and add reduced χ² value
    
    # Uncomment for checking how many alpha values there are.
    #ddofBG = numBins - 2
    #AlphaValue = chi2.sf(redChi2Score*ddofBG, ddofBG)
    #print("α=",AlphaValue)

numBinsChiBG = int((np.max(chi2BG)-np.min(chi2BG))/0.1) # Consider bins of width 0.1
plt.hist(chi2BG,bins = numBinsChiBG,**histStyle) # Plot the distribution of reduced-χ²
plt.xlabel("Reduced Chi Squared Value", **axesFont) # Labelling of axes
plt.xlim([0,5]) # Show the region of the distribution we're most interested in, around 0-5.
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 4B: Distributions of Reduced Chi-Squared Values with "+str(numTrials)+" trials", **titleFont)
#plt.savefig(r'Graphs/Exercise 4B 10000 trials.jpg', dpi=1000)
plt.show()

# In Exercise 4A, we considered whether a background-only scenario would be compatible for the simulated data and found the reduced χ² value to 2.90. When we simulated the distribution of reduced χ² values in 4B, for background-only simulated data using the background-only scenario, we found the majority of reduced χ² values to be centered around 1, with sparingly few lying around the previously-calculated 2.90. Thus, this makes it exceedingly unlikely for the χ² value in 4A to be representative for background-only simulated data. Consequently, we find that it is likely to be false.

# %% 
# Exercise 5A
'''
5. Signal Estimation
Given the background parameterisation, we can test the consistency of a background plus
signal hypothesis. The signal expectation can often be provided by other simulations, as with
the example in the first part of this question.

a. Recalculate the χ2 value for a background + signal hypothesis, presuming the signal
can be characterised by the following parameters: gaus(A_signal = 700, μ = 125 GeV,
s = 1.5 GeV). Comment on the ‘goodness of fit’ and alpha value?
'''
# We chose number of parameters is 2 (A,λ), therefore ddof is 28
# This is because we assume the signal amplitude, mean and standard deviation to be fixed for this hypothesis test. N.B. Reduced Chi-Squared Test is Chi-Squared per Degrees of Freedom
ddof = numBins - 2

## ================================================================
#    NEW, MODIFIED FUNCTIONS FOR THE QUESTION

def get_SB_chi(values, mass_range, nbins, A, lamb, mu, sig, signal_amp):
    bin_heights, bin_edges = np.histogram(values, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_SB_expectation(bin_edges + half_bin_width, newA, newLambda, mu, sig, signal_amp)
    chi = 0

    # Loop over bins - all of them for now. 
    for i in range(len(bin_heights)):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator
    
    return chi/float(ddof)

## ================================================================

signals = np.linspace(histRange[0],histRange[1],10000) # Choose number of signals to plot
A_signal, mu_signal, sig_signal = 700, 125, 1.5 # The question-defined parameters (assumed true)

plt.plot(meanBinValue, vals_bin_heights, 'x', **pointStyle) # Plot histogram points
plt.errorbar(meanBinValue, vals_bin_heights, xerr=widthBin/2,
             yerr=np.sqrt(abs(vals_bin_heights)), **errorStyle)
plt.plot(signals, get_SB_expectation(signals,newA,newLambda,mu_signal,sig_signal,A_signal)) # Plot curve
plt.xlabel("Rest Mass (GeV/c²)", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 5A: Exponential Fit with Gaussian", **titleFont)
#plt.title("Exercise 5A: Exponential-Gaussian Fit \n with " + str(numBins) + " bins, for data in [" + str(histRange[0]) + "," + str(histRange[1]) + "] GeV", **titleFont)
#plt.savefig(r'Graphs/Final Selection/Exercise 5A.jpg', dpi=1000)
plt.show()

newA, newLambda = A, MLE_lambda_new

redChi2SB = get_SB_chi(vals, histRange, numBins, newA, newLambda, mu_signal, sig_signal, A_signal)
print("Exponential-Gaussian Curve: Reduced-χ²=",redChi2SB)

pValue = chi2.sf(redChi2SB*ddof, ddof) # Check value of Alpha
print("Corresponding p-Value, ν=",ddof,", p-value=", pValue)

# %%
# Exercise 5B
'''
b. (*) Using similar techniques to the background parameterisation, justify these signal parameters.
'''
iterations = 20         # Increase number of iterations for improved accuracy.
muRange,sigRange,ampRange = 5, 1, 50 # Specify ranges of values to search from initial guess
mu_signal,sig_signal,amp_signal = 125, 1.5, 700 # The initial guesses provided in 5A
min_chi_SB,min_chi_values_SB = 1e6, [0,0,0] # Storage of these minimised values
# Try for different values of μ, σ and signal amplitude in the specified ranges
mu_sig_test_values = np.linspace(mu_signal-(muRange/2), mu_signal+(muRange/2),iterations)
s_sig_test_values = np.linspace(sig_signal-(sigRange/2), sig_signal+(sigRange/2),iterations)
A_sig_test_values = np.linspace(amp_signal-(ampRange/2),amp_signal+(ampRange/2),iterations)

comboTest = np.zeros([iterations, iterations, iterations, 3])
for i in range(iterations):
    for j in range(iterations):
        for k in range(iterations):
            comboTest[i, j, k] = [mu_sig_test_values[i],s_sig_test_values[j],A_sig_test_values[k]]

chi2TestSignal = np.zeros([iterations, iterations, iterations])
for i in range(iterations):
    for j in range(iterations):
        for k in range(iterations):        
            chi2TestSignal[i,j,k] = get_SB_chi(vals,histRange,numBins,newA,newLambda,comboTest[i,j,k,0],comboTest[i,j,k,1],comboTest[i,j,k,2])
            #print(chi2TestSignal[i,j,k])

            if chi2TestSignal[i,j,k] < min_chi_SB:
                min_chi_SB = chi2TestSignal[i,j,k]
                min_chi_values_SB = [i,j,k]

newMu, newSig, newAmp = mu_sig_test_values[min_chi_values_SB[0]],s_sig_test_values[min_chi_values_SB[1]],A_sig_test_values[min_chi_values_SB[2]]
print(min_chi_SB)
print("We have determined from",iterations,"iterations:\nμ=",newMu,"\nσ=",newSig,"\nA=",newAmp)

curveFit = get_SB_expectation(signals, newA, newLambda, newMu, newSig, newAmp)
plt.plot(meanBinValue, vals_bin_heights, 'x', **pointStyle)
plt.plot(signals, curveFit)
plt.errorbar(meanBinValue, vals_bin_heights, xerr=widthBin/2,
             yerr=np.sqrt(abs(vals_bin_heights)), **errorStyle)
plt.xlabel("Rest Mass (GeV/c²)", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 5B: Exponential-Gaussian Fit \n with " +
          str(numBins) + " bins, for data in [" + str(histRange[0]) + "," + str(histRange[1]) + "] GeV", **titleFont)
plt.savefig(r'Graphs/Exercise 5B.jpg', dpi=1000)
plt.show()

# %% 
# Exercise 5C
'''
c. (**) We tested the background + signal hypothesis for a known signal mass position. In
many searches, the mass isn’t known in advance. Write a program to loop over a range
of masses and plot the corresponding value of the χ^2 of a signal + background
hypothesis (keeping the signal amplitude and gaussian width parameters fixed). How
does the significance of the observed signal at 125 MeV change given we scanned the
whole mass range?
'''

## ================================================================
#    NEW, MODIFIED FUNCTIONS FOR THE QUESTION
ddof = numBins - 5
def get_SB_chi(values, mass_range, nbins, A, lamb, mu, sig, signal_amp):
    bin_heights, bin_edges = np.histogram(values, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_SB_expectation(bin_edges + half_bin_width, newA, newLambda, mu, sig, signal_amp)
    chi = 0

    # Loop over bins - all of them for now. 
    for i in range(len(bin_heights)):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator
    
    return chi/float(ddof)

## ================================================================

iterationsC = 1000        # Increase number of iterations for improved accuracy.
massRange = 50
min_chi_C,min_chi_values_C = 1e6, [0,0,0]
mass_C_test_values = np.linspace(mu_signal-(massRange/2), mu_signal+(massRange/2),iterationsC)

redChi2ValuesC,pValuesC = np.zeros(iterationsC),np.zeros(iterationsC)
for i in range (iterationsC):
    #redChi2ValuesC[i] = get_SB_chi(vals,histRange,numBins,newA,newLambda,mass_C_test_values[i],1.5,700)
    redChi2ValuesC[i] = get_SB_chi(vals,histRange,numBins,newA,newLambda,mass_C_test_values[i],1.5789473684210527,693.421052631579)
    pValuesC[i] = chi2.sf(redChi2ValuesC[i]*ddof, ddof)
    
plt.plot(mass_C_test_values, redChi2ValuesC, **pointStyle)
plt.xlabel("Rest Mass (GeV/c²)", **axesFont)
plt.ylabel("Reduced Chi² Value", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 5C: Distribution of Reduced Chi² Values", **titleFont)
plt.savefig(r'Graphs/Final Selection/Exercise 5C Reduced Chi² Values.jpg', dpi=1000)
plt.show()

plt.plot(mass_C_test_values, pValuesC, **pointStyle)
plt.xlabel("Rest Mass (GeV/c²)", **axesFont)
plt.ylabel("Significance (p-value)", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("Exercise 5C: Distribution of p-values", **titleFont)
plt.savefig(r'Graphs/Final Selection/Exercise 5C p-Values.jpg', dpi=1000)
plt.show()

minIndexC = np.argmin(redChi2ValuesC)
print("Minimum Chi² at mass:",mass_C_test_values[minIndexC],", χ²=",redChi2ValuesC[minIndexC])
maxIndexC = np.argmax(pValuesC)
print("Maximum significance at mass:",mass_C_test_values[maxIndexC],", p-value=",pValuesC[maxIndexC])

print(mass_C_test_values[np.max(np.where(pValuesC > 0.05))])
print(mass_C_test_values[np.min(np.where(pValuesC > 0.05))])

'''
5. Conclusions
We have now tested both a background-only hypothesis, and a background + signal
hypothesis for the simulated dataset. Using your answers to the above, is there sufficient
evidence to claim the existence of a new particle (were this a real experiment of course)?
What could help further build confidence to the claim? So far, we have assumed that statistical
(random) uncertainties have dominated over systematic uncertainties - how might the
evidence change had systematic uncertainties been larger?
'''