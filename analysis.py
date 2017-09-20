#!/usr/bin/python3

"""
Jared Callaham
4/18/17

Data analysis for population annealing trials
- weighted averaging
- calculate thermodynamic pressure estimate and compare to dynamic measurement
- rho_f and rho_t calculation
- plot entropy distributions

"""
import numpy as np
import os

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

Q = 0.95            # "Survival" rate of the population (fixed for this data). In paper, Q=1-epsilon
E = 1.4             # Polydispersity ratio (binary mixture is assumed here)

# Can loop through this to plot multiple trials
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'azure', 'darkgreen', 'chocolate', 'blueviolet']

# Save trial parameters to generalize calculations
trial_params = {90: {"N": 100, "numjobs":986, "R":int(1e6), "label": r"$R=10^6$; $N=100$", "delta_t": 100}, \
                91: {"N": 60, "numjobs":1069, "R":int(1e6), "label": r"$R=10^6$; $N=60$", "delta_t": 100}, \
}

# Output of independent jobs with a trial is stored in different files.
#   This loads all files into a dict that is indexed by job number
def load(trialno):
    data = {}
    for jobno in range(1, trial_params[trialno]["numjobs"]+1):
        filename = "data/{0:03d}/{1:02}.dat".format(trialno, jobno)
        try:
            if os.path.getsize(filename) > 0:
                data[jobno] = np.loadtxt(filename, delimiter='\t', skiprows=1)
        except FileNotFoundError:
            pass
    return data


# Automate labels for plot legend. Looks up in trial_params dict
def make_label(trialno, label_key="label"):
    if label_key == None:
        return ""
    if label_key != "label":
        label = "${0}={1}$".format(label_key, trial_params[trialno][label_key])
    else:
        label = trial_params[trialno][label_key]
    return label


# Return equation of state from population annealing
#   Input a range of packing fractions and an index to that list, and the trial number
#   Output is the equation of state Z at the packing fraction given by that index
def calc_Z(phi_list, step_no, trialno):
    phi = phi_list[step_no]             # Packing fraction given by index
    N = trial_params[trialno]["N"]      # System size
    delta_t = trial_params[trialno]["delta_t"]  # Number of compressions between recorded packing fractions
    rho = phi*12/(np.pi*(1+E**3))       # Number density at this packing fraction
    V = N / rho                         # Extract volume from packing fraction

    # Interpolation for accurate calculation of d_phi/dV
    if (step_no == 0):
        delta_phi = (phi_list[step_no + 1] - phi)
    elif (step_no == len(phi_list) - 1):
        delta_phi = (phi - phi_list[step_no - 1])
    else:
        delta_phi = (phi_list[step_no + 1] - phi_list[step_no - 1]) / 2

    P = rho - (phi*np.log(Q)*delta_t)/(V*delta_phi)   # See Eq (18) in paper
    return P/rho


# Calculate thermodynamic estimate of equation of state. Input trial number and recorded
#   packing fractions. Outputs equation of state estimates and corresponding packing fractions.
def PA_eos(phi, trialno):
    Z = [calc_Z(phi, i, trialno) for i in range(len(phi))]
    return Z, phi


# Isothermal compressibility based on numeric derivative of equation of state
def compressibility(phi, Z, trialno):
    N = trial_params[trialno]["N"]
    r3_avg = (1 + E**3)/2
    rho = phi*6/(np.pi*r3_avg)
    V = N / rho

    # 3-point interpolation to dphi, dZ
    dphi = np.zeros(len(phi))
    dZ = np.zeros(len(phi))
    for i in range(len(phi)):
        if (i == 0):
            dphi[i] = (phi[1] - phi[0])
            dZ[i] = (Z[1] - Z[0])
        elif (i == len(phi) - 1):
            dphi[i] = (phi[i] - phi[i - 1])
            dZ[i] = (Z[i] - Z[i-1])
        else:
            dphi[i] = (phi[i + 1] - phi[i - 1]) / 2
            dZ[i] = (Z[i+1] - Z[i-1])/2

    dP_dV = -6*phi*(phi*dZ/dphi + Z)/(np.pi*V*r3_avg)
    return -1/(V*dP_dV)


# Boublik-Mansoori-Carnahan-Starling-Leland Equation of State
#   Input packing fraction, output Z. Note that this is specific to this binary mixture
def bmcsl(phi):
    # Calculate delta for 50:50 binary mixture
    delta = (np.sqrt(E)*(E-1)**2) / (2 * (1+E**3) )
    y1 = delta * (1+E) / np.sqrt(E)
    y2 = delta * np.sqrt(E) * (1 + E**2) / (1 + E**3)
    y3 = (1 + E**2)**3 / (2 * (1 + E**3)**2 )

    Z = ( (1 + phi + phi**2) - 3*phi*(y1 + y2*phi) - y3*phi**3 ) / (1 - phi)**3
    return Z


# Isothermal compressibility based on analytic derivative of BMCSL equation of state
def bmcsl_compressibility(phi, trialno):
    N = trial_params[trialno]["N"]      # This actually shouldn't depend on N, I think.
    r3_avg = (1 + E**3)/2
    rho = phi*6/(np.pi*r3_avg)
    V = N / rho
    Z = bmcsl(phi)

    # [3.84616 + x (3.55042 - 1.84616 x) ] / (x-1)^4
    y1 = 3.84616
    y2 = 3.55042
    y3 = 1.84616

    dZ_dphi = (y1 + phi*(y2 - y3*phi)) / (phi - 1)**4
    dP_dV = -6*phi*(phi*dZ_dphi + Z)/(np.pi*V*r3_avg)
    return -1/(V*dP_dV) 


# Calculate weighted average of various observables
#   This assumes that the PA used an adaptive schedule, so the proportion of population
#   eliminated in each step is fixed.  See Sec. IID in paper.
def weighted_avg(trialno, phi_interp, pa_pressure=False):
    delta_t = trial_params[trialno]["delta_t"]  # Number of compressions between recorded packing fractions
    R = trial_params[trialno]["R"]  # Population size

    data = load(trialno)        # Load data into a dictionary indexed by job number
    jobs = list(data.keys())

    ##### Interpolate
    interp_data = {}            # Used to save interpolated data from each job before averaging
    for jobno in jobs:
        phi = data[jobno][:-1,0]    # Recorded packing fractions
        Z = data[jobno][:-1,1]      # Z is 1, like/unlike is 3
        if pa_pressure:             # If True, use thermodynamic pressure estimate
            Z, phi = PA_eos(phi, trialno)
        rho_t = data[jobno][:-1,2]
        k = compressibility(phi, Z, trialno)        # Isothermal compressibility
        S = np.log(Q)*np.array(range(len(phi)))*delta_t   # Relative entropies, see Eq (2)
        S += 300  # Add an arbitrary constant to make the floating point work

        # Interpolation, since packing fractions reached aren't identical across trials
        interp_data[jobno] = np.zeros([len(phi_interp), 4])
        interp_data[jobno][:,0] = np.interp(phi_interp, phi, Z)
        interp_data[jobno][:,1] = np.interp(phi_interp, phi, k)
        interp_data[jobno][:,2] = np.interp(phi_interp, phi, rho_t)
        interp_data[jobno][:,3] = np.interp(phi_interp, phi, S)

    # Calculate weights and weighted averages. These arrays will correspond to packing fractions in phi_interp
    Z = np.zeros(len(phi_interp))
    k = np.zeros(len(phi_interp))
    rho_t = np.zeros(len(phi_interp))
    weighted_S = np.zeros(len(phi_interp))
    for i in range(len(phi_interp)):
        norm = np.sum([np.exp(interp_data[jobno][i,3]) for jobno in data.keys() ])   # Denominator in weighted average
        for jobno in jobs:
            S = interp_data[jobno][i, 3]
            weight = np.exp(S) / norm
            # Contribute to weighted averages at this density
            Z[i] += interp_data[jobno][i,0]*weight
            k[i] += interp_data[jobno][i,1]*weight
            rho_t[i] += interp_data[jobno][i, 2]*weight
            weighted_S[i] += np.exp(S)    
        weighted_S[i] = np.log( weighted_S[i] / len(data.keys()) )   # See Eq (5)
        weighted_S -= 300    # Remove earlier constant


    """
    To interpolate rho_f we really want variance in entropy as a function of packing fraction,
        but what we have is essentially variance of packing fraction as a function of entropy.
        So var[S(phi)] ~= (dS/dphi)^2 * var[phi(S)]
    In other words, we have to calculate the variance of packing fractions at each compression step
        and also estimate the slope dS/dphi at each compression step, using (dS/dphi) ~= (dphi/dS)^-1
    """
    # Minimum across jobs of the number of PA compressions (really kmax*delta_t). Used for array sizes
    kmax = 500
    for jobno in jobs:
        kmax = min(kmax, data[jobno].shape[0])

    phi = np.zeros([len(jobs), kmax])
    for i in range(len(jobs)):
        phi[i, :] = np.transpose(data[jobs[i]][:kmax,0])

    var_phi = np.var(phi, 0)        # Now we have variance of packing fraction at each value of entropy

    # Use three point interpolation for derivative of phi with respect to S.
    phi_f = np.mean(phi, 0)         # Estimate center of phi-distributions for each entropy value
    dphi = np.zeros(kmax)
    for i in range(kmax):
        if (i == 0):
            dphi[i] = (phi_f[1] - phi_f[0])
        elif (i == kmax-1):
            dphi[i] = (phi_f[i] - phi_f[i - 1])
        else:
            dphi[i] = (phi_f[i + 1] - phi_f[i-1]) / 2

    dS_dphi = delta_t*np.log(Q)/dphi      # (dS/dphi) ~= (dphi/dS)^-1

    # Now interpolate the variance in packing fraction and the derivative estimate
    var_phi = np.interp(phi_interp, phi_f, var_phi)
    dS_dphi = np.interp(phi_interp, phi_f, dS_dphi)

    # And estimate rho_f
    rho_f = R*(dS_dphi)**2*var_phi

    return phi_interp, Z, k, rho_t, rho_f, weighted_S

######################################
# Calculate and plot observables
######################################

# Figures:
#   1 - Equation of state
#   2 - Difference between equation of state and BMCSL prediction
#   3 - Difference between thermodynamic and dynamic estimates of Z
#   4 - rho_f and rho_t
#   5 - Isothermal compressibility


phi_interp = np.linspace(0, 0.62, num=1000)     # To interpolate between different phi values across trials

# Plot empirical BMCSL equations
plt.figure(1)
plt.plot(phi_interp, bmcsl(phi_interp), color='k', label="BMCSL Equation of State")

plt.figure(5)
plt.plot(phi_interp, bmcsl_compressibility(phi_interp, 91), color='k', label="BMCSL Equation of State")

color_counter=0
for trialno in [90, 91]:
    phi, Z, k, rho_t, rho_f, _ = weighted_avg(trialno, phi_interp, pa_pressure=False)
    _, PA_Z, _, _, _, _ = weighted_avg(trialno, phi_interp, pa_pressure=True)

    plt.figure(1)
    plt.plot(phi, Z, color=colors[color_counter], label=make_label(trialno))

    plt.figure(2)
    plt.plot(phi, Z-bmcsl(phi_interp), color=colors[color_counter], label=make_label(trialno))

    plt.figure(3)
    plt.plot(phi, (PA_Z-Z)/Z, color=colors[color_counter], label=make_label(trialno))

    plt.figure(4)
    plt.semilogy(phi, rho_f, color=colors[color_counter], label=make_label(trialno, "N")+r': $\rho_f$')
    plt.semilogy(phi, rho_t, color=colors[color_counter], ls='--', label=make_label(trialno, "N")+r': $\rho_t$')

    plt.figure(5)
    plt.plot(phi, k, color=colors[color_counter], label=make_label(trialno))

    color_counter += 1

# Format plots
plt.figure(1)
plt.legend(loc=2)
plt.xlabel("Packing fraction " + r"$\varphi$")
plt.ylabel("Equation of state $Z$")
plt.xlim([0.55,0.62])
plt.ylim([10, 40])
plt.tight_layout()


plt.figure(2)
plt.legend(loc=2)
plt.xlabel("Packing fraction " + r"$\varphi$")
plt.ylabel("$Z_\mathrm{ECMC} - Z_\mathrm{BMCSL}$")
plt.xlim([0.55,0.62])
plt.ylim([0, 8])
plt.tight_layout()

plt.figure(3)
plt.legend(loc=2)
plt.xlabel("Packing fraction " + r"$\varphi$")
plt.ylabel("$\Delta Z / Z$")
plt.xlim([0.55,0.62])
plt.ylim([-.001, .001])
plt.tight_layout()

plt.figure(4)
plt.legend(loc=2)
plt.xlabel("Packing fraction " + r"$\varphi$")
plt.ylabel(r"$\rho_t$, $\rho_f$")
plt.xlim([.5, .62])
plt.ylim([1e2, 1e7])
plt.tight_layout()

plt.figure(5)
plt.legend(loc=2)
plt.xlabel("Packing fraction " + r"$\varphi$")
plt.ylabel("Isothermal Compressibility " + r"$k_T$")
plt.xlim([.5, .62])
plt.ylim([0, 0.04])
plt.tight_layout()

plt.show()


######################################################
# Plot joint distributions of pressure and entropy
##################################################
trialno = 90
data = load(trialno)
delta_t = trial_params[trialno]["delta_t"]
for phi_i in [0.56, 0.58, 0.6, 0.62]:
    entropy = []
    pressure = []
    for jobno in data.keys():
        phi = data[jobno][:, 0]
        Z = data[jobno][:, 1]
        S = np.log(Q)*np.array(range(len(phi)))*delta_t  # Relative entropies, see Eq (2)

        entropy.append(np.interp(phi_i, phi, S))
        pressure.append(np.interp(phi_i, phi, Z))

    plt.scatter(entropy, pressure)

    # Also plot weighted average as larger point
    _, Z, _, _, _, S = weighted_avg(trialno, [phi_i])
    plt.scatter(S, Z, color='r', marker='s', s=40)

    plt.xlabel('Relative Entropy')
    plt.ylabel('Equation of State')
    plt.title('{0:0.2f}'.format(phi_i))
    plt.show()



########################################################################
# Perform bootstrap of weighted averages to estimate errors and rho_star
#   Note: comment out if bootstrap already done - plotting functions load from data file
########################################################################

"""
# Bootstrap observables
def bootstrap(trialno, phi_interp, num_boots):
    delta_t = trial_params[trialno]["delta_t"]      # Number of compressions between measurement
    numjobs = trial_params[trialno]["numjobs"]      # Number of independent runs in this trial

    data = load(trialno)

    # Precalculate interpolations and weights, so bootstrapping just involves normalizing and resumming
    interp_Z = np.zeros([numjobs, len(phi_interp)])     # Equation of state
    interp_S = np.zeros([numjobs, len(phi_interp)])     # Entropy
    interp_k = np.zeros([numjobs, len(phi_interp)])     # 
    weight = np.zeros([numjobs, len(phi_interp)])       # Weight for averaging (e^S)
    
    # Assume that trials are sequentially numbered from (1, M)
    #       -> This may require some processing
    for i in range(numjobs):
        jobno = i+1
        phi = data[jobno][:-1,0]    # Packing fraction
        Z = data[jobno][:-1,1]      # Dynamic equation of state
        k_T = compressibility(phi, Z, trialno)  # Isothermal compressibility from numeric derivative of Z

        S = np.log(Q)*np.array(range(len(phi)))*delta_t   # Relative entropies, see Eq (2)
        S += 300  # Add an arbitrary constant to make the floating point work

        interp_Z[i, :] = np.interp(phi_interp, phi, Z)
        interp_S[i, :] = np.interp(phi_interp, phi, S)
        interp_k[i, :] = np.interp(phi_interp, phi, k_T)

    weight = np.exp(interp_S)

    # Arrays for storing bootstrapped weighted averages
    boot_k = np.zeros([num_boots, len(phi_interp)])         # Isothermal compressibility
    boot_eos = np.zeros([num_boots, len(phi_interp)])       # Dynamic equation of state
    boot_S = np.zeros([num_boots, len(phi_interp)])         # Weighted average entropy

    # Now bootstrap by randomly resampling runs and resumming weighted average
    for iteration in range(num_boots):
        print("Bootstrap {0}".format(iteration))
        # Construct the resampled population
        boot_trials = np.random.choice(numjobs, numjobs, True)

        # Calculate weights and weighted averages
        for i in range(len(phi_interp)):
            norm = np.sum([weight[jobno, i] for jobno in boot_trials])
            for j in range(numjobs):
                jobno = boot_trials[j]
                W = weight[jobno, i] / norm
                boot_S[iteration, i] += weight[jobno, i]
                boot_k[iteration, i] += interp_k[jobno, i]*W
                boot_eos[iteration, i] += interp_Z[jobno, i]*W

            boot_S[iteration, i] = np.log( boot_S[iteration, i] / numjobs ) - 300    # Remove earlier constant

    # Construct 95% confidence interval from bootstrapped data
    cutoff = .025*num_boots

    k_ci = np.zeros([len(phi_interp), 2])
    boot_k = np.sort(boot_k, 0)
    k_ci[:, 0] = np.transpose(boot_k[cutoff-1, :])
    k_ci[:, 1] = np.transpose(boot_k[-cutoff, :])

    Z_ci = np.zeros([len(phi_interp), 2])
    boot_eos = np.sort(boot_eos, 0)
    Z_ci[:, 0] = np.transpose(boot_eos[cutoff-1, :])
    Z_ci[:, 1] = np.transpose(boot_eos[-cutoff, :])

    # Look at variance of weighted average entropy (for calculating rho_star)
    var_S = np.var(boot_S, 0)
        
    return Z_ci, k_ci, var_S


# Perform bootstrapping
print("Bootstrapping...")
for trialno in [90, 91]:
    phi, Z, k, _, _, _ = weighted_avg(trialno, phi_interp, pa_pressure=False)   # Normal weighted average
    Z_ci, k_ci, var_S = bootstrap(trialno, phi_interp, num_boots=10000)      # Confidence interval from bootstrapping

    # Save results
    data = np.zeros([len(phi), 8])
    data[:, 0] = phi
    data[:, 1] = Z
    data[:, 2:4] = Z_ci
    data[:, 4] = k
    data[:, 5:7] = k_ci
    data[:, 7] = var_S

    header = "Phi\tZ\tlower\tupper\tk\tlower\tupper\tvar_S"
    np.savetxt("data/{0:03d}/bootstrapped.dat".format(trialno), data, delimiter='\t', header=header)
"""



###########################################################################
# Load bootstrapped results and plot observables with confidence intervals
############################################################################

# Figures:
# 1 - Dynamic EOS
# 2 - Difference between dynamic EOS and BMCSL prediction
# 3 - Isothermal compressibility
# 4 - rho_f and rho_star vs packing fraction


phi_interp = np.linspace(0, 0.62, num=1000)     # To interpolate between different phi values across trials

plt.figure(1)
plt.plot(phi_interp, bmcsl(phi_interp), color='k', label="BMCSL Equation of State")

plt.figure(3)
plt.plot(phi_interp, bmcsl_compressibility(phi_interp, 91), color='k', label="BMCSL Equation of State")


color_counter=0
for trialno in [90, 91]:
    data = np.loadtxt("data/{0:03d}/bootstrapped.dat".format(trialno), delimiter="\t", skiprows=1)
    numjobs = trial_params[trialno]["numjobs"]          # Number of independent runs
    R = trial_params[trialno]["R"]                      # Population size

    # Dynamic equation of state
    plt.figure(1)
    phi = data[:, 0]
    Z = data[:, 1]
    lower = data[:, 2]
    upper = data[:, 3]
    plt.plot(phi, Z, color=colors[color_counter], label=make_label(trialno))
    plt.fill_between(phi, lower, upper, facecolor=colors[color_counter], alpha=0.4, interpolate=True)

    # Difference between dynamic equation of state and BMCSL
    plt.figure(2)
    plt.plot(phi, Z-bmcsl(phi), color=colors[color_counter], label=make_label(trialno))
    plt.fill_between(phi, lower-bmcsl(phi), upper-bmcsl(phi), facecolor=colors[color_counter], alpha=0.4, interpolate=True)

    # Isothermal compressibility
    plt.figure(3)
    phi = data[:, 0]
    k = data[:, 4]
    lower = data[:, 5]
    upper = data[:, 6]
    plt.plot(phi, k, color=colors[color_counter], label=make_label(trialno))
    plt.fill_between(phi, lower, upper, facecolor=colors[color_counter], alpha=0.4, interpolate=True)

    # rho_f and rho_star
    plt.figure(4)
    phi = data[:, 0]
    var_S = data[:, 7]
    rho_star = numjobs*R*var_S      # Eq (8)
    plt.semilogy(phi, rho_star, label=make_label(trialno, "N") + r": $\rho_f^*$", ls='--', color=colors[color_counter])

    phi, _, _, _, rho_f, _ = weighted_avg(trialno, phi_interp, pa_pressure=False)
    plt.semilogy(phi, rho_f, label=make_label(trialno, "N") + r": $\rho_f$", color=colors[color_counter])

    color_counter += 1

# Format and show figures
plt.figure(1)
plt.legend(loc=2)
plt.xlim([.55, .62])
plt.ylim([15, 40])
plt.xlabel('Packing Fraction')
plt.ylabel('Equation of State')
plt.tight_layout()


plt.figure(2)
plt.legend()
plt.xlabel(r'$\mathrm{Packing Fraction} \; \varphi$')
plt.ylabel(r'$Z_\mathrm{ECMC} - Z_\mathrm{BMCSL}$')
plt.xlim([0.55, 0.6])
plt.ylim([0, 2.5])
plt.tight_layout()

plt.figure(3)
plt.legend()
plt.xlabel('Packing Fraction')
plt.ylabel('Isothermal Compressibility')
plt.xlim([0.5, 0.62])
plt.ylim([0, 0.04])
plt.tight_layout()

plt.figure(4)
plt.legend(loc=2)
plt.xlabel(r'$\mathrm{Packing Fraction} \; \varphi$')
plt.ylabel(r'$\rho_f, \; \rho_f^*$') 
plt.xlim([.5, .62])
plt.ylim([1e2, 1e9])
plt.tight_layout()

plt.show() 



