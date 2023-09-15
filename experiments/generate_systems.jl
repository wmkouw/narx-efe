using JLD
using DSP
using Distributions
using ProgressMeter
using LinearAlgebra

# Experimental parameters
num_exps = 100
N = 100
fs = 20 # Hertz
Δt = 1/fs

# Basis
H = 3
sys_basis(x) = cat([1.0; [x.^d for d in 1:H]]...,dims=1)
M_in = 3
M_out = 3
M = size(sys_basis(zeros(M_out + 1 + M_in)),1)

# Randomized variables
pσ = Gamma(2.0, 1/100.)
pf = Uniform(1.0, 1.5)

ppp = Progress(num_exps)
for nn in 1:num_exps

    # Define system parameters
    sys_mnoise_sd = rand(pσ)
    sys_lowpass_f = rand(pf)
    df = digitalfilter(Lowpass(sys_lowpass_f; fs=fs), Butterworth(maximum([M_in, M_out])))
    sys_coefficients = [0.0; sys_basis([coefb(df)[2:M_out+1]; coefa(df)[1:M_in+1]])[2:end]]

    save("./experiments/data/system-$nn.jld", "sys_sd", sys_mnoise_sd, "sys_theta", sys_coefficients)

    next!(ppp)
end
finish!(ppp)