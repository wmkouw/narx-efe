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
using Plots; default(grid=false, label="", linewidth=3,margin=20Plots.pt)

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
tN = range(0.0, step=Δt, length=N)
```

```julia
# Inputs
A  = rand(10)*200 .- 100
Ω  = rand(10)*3
controls = mean([A[i]*sin.(Ω[i].*tN) for i = 1:10]) ./ 10;
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
plot(xlabel="time (steps)", size=(900,300))
plot!(tN, controls, color="red", label="controls")
plot!(tN, observations, color="black", label="observations")
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
qθ = [pθ0]
qτ = [pτ0]
FE = zeros(10,N)

T = 1
preds = (zeros(N,T), zeros(N,T))

@showprogress for k in 1:N
    
    # Make predictions
    preds[1][k,:], preds[2][k,:] = predictions(agent, controls[k], time_horizon=T)
    
    # Update beliefs
    NARXAgents.update!(agent, observations[k], controls[k])
    FE[:,k] = agent.free_energy
    
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
    pwm = plot(ylims=(-1.,1.))
    
    hline!([sys_coefficients[m]], color="black", label="true")
    plot!(mθ[m,:], ribbon=sqrt.(vθ[m,:]), color="purple", label="belief", ylabel="θ_$m")
    
    push!(pw,pwm)
end
plot(pw..., layout=(4,1), size=(600,1200))
```

```julia
plot(FE[:], ylabel="F[q]")
```

```julia
p1 = plot(xlabel="time [steps]", title="Observations vs 1-step ahead predictions")
scatter!(observations[2:end], color="black", label="observations")
plot!(preds[1][2:end,1], ribbon=preds[2][2:end,1], color="purple", label="k=$T prediction")
```

## Expected Free Energy minimization

```julia
# Length of trial
N = 100
tN = 1:N
T = 3;

# Set control properties
m_star = 1.0
v_star = 1e-3
goal = NormalMeanVariance(m_star,v_star)
control_prior = 1e-2
num_iters = 10

# Initialize beliefs
pτ = [pτ0]
pθ = [pθ0]

# Start system
system = ARXsys(sys_coefficients, 
                sys_mnoise_sd, 
                order_outputs=M_out, 
                order_inputs=M_in)

# Start agent
agent = NARXAgent(pθ0, pτ0, 
                  goal_prior=goal, 
                  memory_actions=Lu, 
                  memory_senses=Ly, 
                  pol_degree=H,
                  thorizon=T,
                  control_prior=control_prior,
                  num_iters=num_iters)

# Preallocate
y_ = zeros(N)
u_ = zeros(N+1)

pred_m = zeros(N,T)
pred_v = zeros(N,T)
FE = zeros(num_iters, N)

@showprogress for k in 1:N
    
    # Act upon environment
    ARXsystem.update!(system, u_[k])
    y_[k] = system.observation
    
    # Update parameter beliefs
    NARXAgents.update!(agent, y_[k], u_[k])
    
    FE[:,k] = agent.free_energy
    push!(pθ, agent.qθ)
    push!(pτ, agent.qτ)
    
    # Optimal control
    policy = minimizeEFE(agent)
    u_[k+1] = policy[1]
    
    # Store future predictions
    pred_m[k,:], pred_v[k,:] = predictions(agent, policy, time_horizon=T)
    
end
```

```julia
plot(FE[end,:] .- FE[1,:])
```

```julia
p1 = plot(tN, y_, color="black", label="observations")
hline!([m_star], color="green", ylims=[-1., 2.])
p4 = plot(tN, u_[1:end-1], color="red", ylabel="controls", xlabel="time [s]")

plot(p1,p4, layout=grid(2,1, heights=[.7, .3]), size=(900,400))
```

```julia
tSθ = [tr(cov(pθ[k])) for k in 1:N]
plot(tSθ)
```

```julia
limsb = [minimum(y_)*1.5, maximum(y_)*1.5]

window = 20

anim = @animate for k in 2:2:(N-T-1)
    
    if k <= window
        plot(tN[1:k], y_[1:k], color="blue", xlims=(tN[1], tN[window+T+1]+0.5), label="past data", xlabel="time (sec)", ylims=limsb, size=(900,300))
        plot!(tN[k:k+T], y_[k:k+T], color="purple", label="true future", linestyle=:dot)
        plot!(tN[k+1:k+T], pred_m[k,:], ribbon=pred_v[k,:], label="predicted future", color="orange", legend=:topleft)
        hline!([m_star], color="green")
    else
        plot(tN[k-window:k], y_[k-window:k], color="blue", xlims=(tN[k-window], tN[k+T+1]+0.5), label="past data", xlabel="time (sec)", ylims=limsb, size=(900,300))
        plot!(tN[k:k+T], y_[k:k+T], color="purple", label="true future", linestyle=:dot)
        plot!(tN[k+1:k+T], pred_m[k,:], ribbon=pred_v[k,:], label="prediction", color="orange", legend=:topleft)
        hline!([m_star], color="green")
    end
    
end
gif(anim, "figures/NARX-EFE-verification-plan_trial00.gif", fps=24)
```

```julia

```
