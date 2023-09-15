using Revise
using FFTW
using DSP
using SpecialFunctions
using LinearAlgebra
using ProgressMeter
using Distributions
using JLD

includet("../NARXAgents.jl"); using .NARXAgents
includet("../NARXsystem.jl"); using .NARXsystem
includet("../mv_normal_gamma.jl"); using .mv_normal_gamma 

# Experimental parameters
num_exps = 100
N = 100
fs = 20 # Hertz
Δt = 1/fs
T = 10

# Basis
H = 3
sys_basis(x) = cat([1.0; [x.^d for d in 1:H]]...,dims=1)
M_in = 2
M_out = 2
M = size(sys_basis(zeros(M_out + 1 + M_in)),1)

# Control parameters
input_lims = (-1.,1.)

# Specify prior distributions
α0 = 10.0
β0 = 1.0
μ0 = zeros(M)
Λ0 = diagm(ones(M))
goal = NormalMeanVariance(1.0, 1.0)

# Randomized variables
pσ = Gamma(2.0, 1/50.)
pf = Uniform(0.5, 2.0)

@showprogress for nn in 1:num_exps

    # Define system parameters
    sys_mnoise_sd = rand(pσ);
    df = digitalfilter(Lowpass(rand(pf); fs=fs), Butterworth(maximum([M_in, M_out])))
    sys_coefficients = [0.0; sys_basis([coefb(df)[2:M_out+1]; coefa(df)[1:M_in+1]])[2:end]]

    # Inputs
    controls = clamp!(2randn(N).-1, input_lims...)

    # Outputs
    system = NARXsys(sys_coefficients, 
                    sys_basis, 
                    sys_mnoise_sd, 
                    order_outputs=M_out, 
                    order_inputs=M_in, 
                    input_lims=input_lims)

    py = []
    μ = [μ0]
    Λ = [Λ0]
    α = [α0]
    β = [β0]
    FE = zeros(N)

    agent = NARXAgent(μ0, Λ0, α0, β0,
                      goal_prior=goal, 
                      delay_inp=M_in, 
                      delay_out=M_out, 
                      pol_degree=H)

    outputs = zeros(N)
    inputs = zeros(N)
    inputs_ = [inputs; zeros(T)]

    for k in 1:N

        # Evolve system
        NARXsystem.update!(system, controls[k])
        outputs[k] = system.observation
        inputs[k] = system.input_buffer[1]
        
        # Make predictions
        push!(py, predictions(agent, inputs_[k:k+T], time_horizon=T))
        
        # Update beliefs
        NARXAgents.update!(agent, outputs[k], inputs[k])
        
        push!( μ, agent.μ )
        push!( Λ, agent.Λ )
        push!( α, agent.α )
        push!( β, agent.β )

        FE[k] = agent.free_energy
    end

    @save "results/learningrate-rnd-$nn.jl" py, μ, Λ, α, β, FE, α0, β0, μ0, Λ0, goal, pσ, pf, input_lims, T

end
