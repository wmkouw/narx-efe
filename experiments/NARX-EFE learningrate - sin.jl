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
tsteps = collect(range(0.0, step=Δt, length=N))
T = 10

# Basis
H = 3
sys_basis(x) = cat([1.0; [x.^d for d in 1:H]]...,dims=1)
M_in = 3
M_out = 3
M = size(sys_basis(zeros(M_out + 1 + M_in)),1)

# Control parameters
input_lims = (-1.,1.)

# Specify prior distributions
α0 = 10.0
β0 = 1.0
μ0 = zeros(M)
Λ0 = diagm(ones(M))
goal = Normal(1.0, 1.0)

ppp = Progress(num_exps)
for nn in 1:num_exps

    # Load system parameters
    sys_settings = load("./experiments/data/system-$nn.jld")
    sys_mnoise_sd = sys_settings["sys_sd"]
    sys_coefficients = sys_settings["sys_theta"]

    # Sinusoidal controls
    Ω  = rand(5)*10
    controls = mean([sin.(ω.*tsteps) for ω in Ω]);

    # Start system
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

    save("./experiments/results/learningrate-sin-$nn.jld", "py", py, "mu", μ, "Lambda", Λ, "alpha", α, "beta", β, 
        "FE", FE, "alpha0", α0, "beta0", β0, "mu0", μ0, "Lambda0", Λ0, "goal", goal, "u_lims", input_lims, "thorizon", T)

    next!(ppp)
end
finish!(ppp)
