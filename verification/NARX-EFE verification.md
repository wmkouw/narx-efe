---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: Julia 1.9.1
    language: julia
    name: julia-1.9
---

# Bayesian black-box model-predictive control

An Expected Free Energy minimizing agent based on a nonlinear autoregressive model with exogenous input.


## System: ARX

Consider a system generating observations $y_k \in \mathbb{R}$ based on inputs $u_k \in \mathbb{R}$:

$$ y_k = \theta^{\top}x_k + u_k + e_k \, ,$$

where $x_k \in \mathbb{R}^{D}$ is a vector of previous observations and $\theta$ are coefficients. The noise instances $e_k$ are assumed to be zero-mean Gaussian distributed with variance $\sigma^2$.

```julia
using Revise
using ForwardDiff
using Optim
using RxInfer
using SpecialFunctions
using LinearAlgebra
using ProgressMeter
using Distributions
using Plots; default(grid=false, label="", linewidth=3,margin=10Plots.pt)

includet("../NARXAgents.jl"); using .NARXAgents
includet("./ARXsystem.jl"); using .ARXsystem
```

```julia
# Noise
sys_mnoise_sd = 1e-1;

# True coefficients
M_in = 1
M_out = 2
M = M_in + M_out
sys_coefficients = [0.0; rand(M) .- 0.5]
```

```julia
# Time
N = 300
Δt = 0.05
time = range(0.0, step=Δt, length=N);
```

```julia
# Inputs
A  = rand(10)*200 .- 100
Ω  = rand(10)*3
controls = mean([A[i]*sin.(Ω[i].*time) for i = 1:10]) ./ 10;
```

```julia
# Outputs

system = ARXsys(sys_coefficients, sys_mnoise_sd, order_outputs=M_out, order_inputs=M_in)

observations = zeros(N)
for k in 1:N
    ARXsystem.update!(system, controls[k])
    observations[k] = system.observation
end
```

```julia
plot(xlabel="time [s]", size=(900,300))
plot!(time, controls, color="red", label="controls")
plot!(time, observations, color="black", label="observations")
```

## NARX model

```julia
# Polynomial degree
H = 1

# Delay order
Ly = 2
Lu = 1

# Model order
M = size(ϕ(zeros(Ly+Lu), degree=H),1);
```

```julia
# Specify prior distributions
pτ0 = GammaShapeRate(1e-1, 1e-1)
pθ0 = MvNormalMeanCovariance(ones(M), 1e2diagm(ones(M)))
```

```julia
agent = NARXAgent(pθ0, pτ0, memory_actions=Lu, memory_senses=Ly, pol_degree=H)
```

## Parameter estimation

```julia
qθ = []
qτ = []

@showprogress for (k,t) in enumerate(time)
    
    NARXAgents.update!(agent, observations[k], controls[k])

    # Update beliefs
    push!(qθ, agent.qθ)
    push!(qτ, agent.qτ)

end
```

```julia
mθ = cat(mean.(qθ)...,dims=2)
vθ = cat( var.(qθ)...,dims=2)
```

```julia
pw = []
for m in 1:M
    pwm = plot(xlabel="time (s)",ylims=(-1.,1.), size=(900,600))
    
    hline!([sys_coefficients[m]], color="black")
    plot!(time, mθ[m,:], ribbon=sqrt.(vθ[m,:]), color="purple")
    
    push!(pw,pwm)
end
plot(pw..., layout=(4,1), size=(600,1200))
```
