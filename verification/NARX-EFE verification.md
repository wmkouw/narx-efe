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
sys_mnoise_sd = 0.05;

# True coefficients
M_in = 2
M_out = 2
M = M_in + M_out
sys_coefficients = [0.0; randn(M)./3]
```

```julia
# Time
N = 100
Δt = 0.1
tsteps = collect(range(0.0, step=Δt, length=N));
```

```julia
# Inputs
A  = rand(10)*200 .- 100
Ω  = rand(10)*3
controls = mean([A[i]*sin.(Ω[i].*tsteps
) for i = 1:10]) ./ 10;
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
plot!(tsteps, controls, color="red", label="controls")
plot!(tsteps, observations, color="black", label="observations")
```

## NARX model

```julia
# Polynomial degree
H = 1

# Delay order
Ly = 2
Lu = 2

# Model order
M = size(ϕ(zeros(Ly+Lu), degree=H),1);
```

```julia
# Specify prior distributions
pτ0 = GammaShapeRate(1e0, 1e0)
pθ0 = MvNormalMeanCovariance(ones(M), 1e3diagm(ones(M)))
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

@showprogress for k in 1:(N-T)
    
    # Make predictions
    preds[1][k,:], preds[2][k,:] = predictions(agent, controls[k:k+T], time_horizon=T)
    
    # Update beliefs
    NARXAgents.update!(agent, observations[k], controls[k])
    FE[:,k] = agent.free_energy
    
    push!(qθ, agent.qθ)
    push!(qτ, agent.qτ)

end
```

```julia
plot(FE[:], xlabel="updates (time x num_iters)", ylabel="F[q]")
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
plot(pw..., layout=(M,1), size=(600,1200))
```

```julia
limsb = [minimum(preds[1])*1.2, maximum(preds[1])*1.2]
K = 1

p1 = plot(xlabel="time [steps]", title="Observations vs $K-step ahead predictions", ylims=limsb)
scatter!(observations[2:end], color="black", label="observations")
plot!(preds[1][2:end,K], ribbon=preds[2][2:end,K], color="purple", label="k=$K prediction")
```

## Experiments

```julia
# Length of trial
N = 100
tsteps = range(0.0, step=Δt, length=N)
T = 10

# Set control properties
goal = NormalMeanVariance(1.0, 1e-2)
control_prior = 0.0
num_iters = 10
u_lims = (-15, 15)
tlimit = 300

# Specify prior distributions
pτ0 = GammaShapeRate(1e0, 1e0)
pθ0 = MvNormalMeanCovariance(ones(M), 1e3diagm(ones(M)))
```

### Mean Squared Error minimization

```julia
# Start system
system = ARXsys(sys_coefficients, 
                sys_mnoise_sd, 
                order_outputs=M_out, 
                order_inputs=M_in)
```

```julia
# Initialize beliefs
pτ_MSE = [pτ0]
pθ_MSE = [pθ0]

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
y_MSE = zeros(N)
u_MSE = zeros(N+1)

pred_m = zeros(N,T)
pred_v = zeros(N,T)
FE_MSE = zeros(num_iters, N)

@showprogress for k in 1:N
    
    # Act upon environment
    ARXsystem.update!(system, u_MSE[k])
    y_MSE[k] = system.observation
    
    # Update parameter beliefs
    NARXAgents.update!(agent, y_MSE[k], u_MSE[k])
    
    FE_MSE[:,k] = agent.free_energy
    push!(pθ_MSE, agent.qθ)
    push!(pτ_MSE, agent.qτ)
    
    # Optimal control
    policy = minimizeMSE(agent)
    u_MSE[k+1] = policy[1]
    
    # Store future predictions
    pred_m[k,:], pred_v[k,:] = predictions(agent, policy, time_horizon=T)
    
end
```

```julia
plot(FE_MSE[:], xlabel="updates (time x num_iters)", ylabel="F[q]")
```

```julia
p1 = plot(tsteps, y_MSE, color="black", label="observations")
hline!([mean(goal)], color="green", label="goal")
p4 = plot(tsteps, u_MSE[1:end-1], color="red", ylabel="controls", xlabel="time [s]")

plot(p1,p4, layout=grid(2,1, heights=[.7, .3]), size=(900,400))
```

```julia
savefig("figures/NARX-MSE-verification-trial.png")
```

```julia
dSθ_MSE = [det(cov(pθ_MSE[k])) for k in 1:N]
final_dSθ_MSE = dSθ_MSE[end]
plot(dSθ_MSE, title="|Σ| = $final_dSθ_MSE", yscale=:log10)
```

```julia
K = 10
sum_vy_k = sum(pred_v[:,K])
plot(pred_v[:,K], title="Sum V[y_t+$K] = $sum_vy_k", yscale=:log10)
```

```julia
limsb = [minimum(y_MSE)*1.5, maximum(y_MSE)*1.5]

window = 20

anim = @animate for k in 2:2:(N-T-1)
    
    if k <= window
        plot(tsteps[1:k], y_MSE[1:k], color="blue", xlims=(tsteps[1], tsteps[window+T+1]+0.5), label="past data", xlabel="time (sec)", ylims=limsb, size=(900,300))
        plot!(tsteps[k:k+T], y_MSE[k:k+T], color="purple", label="true future", linestyle=:dot)
        plot!(tsteps[k+1:k+T], pred_m[k,:], ribbon=pred_v[k,:], label="predicted future", color="orange", legend=:topleft)
        hline!([mean(goal)], color="green")
    else
        plot(tsteps[k-window:k], y_MSE[k-window:k], color="blue", xlims=(tsteps[k-window], tsteps[k+T+1]+0.5), label="past data", xlabel="time (sec)", ylims=limsb, size=(900,300))
        plot!(tsteps[k:k+T], y_MSE[k:k+T], color="purple", label="true future", linestyle=:dot)
        plot!(tsteps[k+1:k+T], pred_m[k,:], ribbon=pred_v[k,:], label="prediction", color="orange", legend=:topleft)
        hline!([mean(goal)], color="green")
    end
    
end
gif(anim, "figures/NARX-MSE-verification-planning.gif", fps=24)
```

### Expected Free Energy minimization

```julia
# Start system
system = ARXsys(sys_coefficients, 
                sys_mnoise_sd, 
                order_outputs=M_out, 
                order_inputs=M_in)
```

```julia
# Initialize beliefs
pτ_EFE = [pτ0]
pθ_EFE = [pθ0]

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
y_EFE = zeros(N)
u_EFE = zeros(N+1)

pred_m = zeros(N,T)
pred_v = zeros(N,T)
FE_EFE = zeros(num_iters, N)

@showprogress for k in 1:N
    
    # Act upon environment
    ARXsystem.update!(system, u_EFE[k])
    y_EFE[k] = system.observation
    
    # Update parameter beliefs
    NARXAgents.update!(agent, y_EFE[k], u_EFE[k])
    
    FE_EFE[:,k] = agent.free_energy
    push!(pθ_EFE, agent.qθ)
    push!(pτ_EFE, agent.qτ)
    
    # Optimal control
    policy = minimizeEFE(agent)
    u_EFE[k+1] = policy[1]
    
    # Store future predictions
    pred_m[k,:], pred_v[k,:] = predictions(agent, policy, time_horizon=T)
    
end
```

```julia
plot(FE_EFE[:], xlabel="updates (time x num_iters)", ylabel="F[q]")
```

```julia
p1 = plot(tsteps, y_EFE, color="black", label="observations")
hline!([mean(goal)], color="green", label="goal")
p4 = plot(tsteps, u_EFE[1:end-1], color="red", ylabel="controls", xlabel="time [s]")

plot(p1,p4, layout=grid(2,1, heights=[.7, .3]), size=(900,400))
```

```julia
savefig("figures/NARX-EFE-verification-trial.png")
```

```julia
dSθ_EFE = [det(cov(pθ_EFE[k])) for k in 1:N]
final_dSθ_EFE = dSθ_EFE[end]
plot(dSθ_EFE, title="|Σ| = $final_dSθ_EFE", yscale=:log10)
```

```julia
K = 10
sum_vy_k = sum(pred_v[:,K])
plot(pred_v[:,K], title="Sum V[y_t+$K] = $sum_vy_k", yscale=:log10)
```

```julia
limsb = [minimum(y_EFE)*1.5, maximum(y_EFE)*1.5]

window = 20

anim = @animate for k in 2:2:(N-T-1)
    
    if k <= window
        plot(tsteps[1:k], y_EFE[1:k], color="blue", xlims=(tsteps[1], tsteps[window+T+1]+0.5), label="past data", xlabel="time (sec)", ylims=limsb, size=(900,300))
        plot!(tsteps[k:k+T], y_EFE[k:k+T], color="purple", label="true future", linestyle=:dot)
        plot!(tsteps[k+1:k+T], pred_m[k,:], ribbon=pred_v[k,:], label="predicted future", color="orange", legend=:topleft)
        hline!([mean(goal)], color="green")
    else
        plot(tsteps[k-window:k], y_EFE[k-window:k], color="blue", xlims=(tsteps[k-window], tsteps[k+T+1]+0.5), label="past data", xlabel="time (sec)", ylims=limsb, size=(900,300))
        plot!(tsteps[k:k+T], y_EFE[k:k+T], color="purple", label="true future", linestyle=:dot)
        plot!(tsteps[k+1:k+T], pred_m[k,:], ribbon=pred_v[k,:], label="prediction", color="orange", legend=:topleft)
        hline!([mean(goal)], color="green")
    end
    
end
gif(anim, "figures/NARX-EFE-verification-plan_trial00.gif", fps=24)
```

### Comparison

```julia
println("Final |Σ| MSE = $final_dSθ_MSE")
println("Final |Σ| EFE = $final_dSθ_EFE")
```

```julia
idθ_MSE = [-logpdf(pθ_MSE[k], sys_coefficients) for k in 1:N]
idθ_EFE = [-logpdf(pθ_EFE[k], sys_coefficients) for k in 1:N]

plot(xlabel="time (steps)", ylabel="-logpdf θ*", size=(900,400))
plot!(idθ_MSE, label="MSE")
plot!(idθ_EFE, label="EFE")
```

```julia
idθ_MSE = [-logpdf(pθ_MSE[k], sys_coefficients) for k in 1:N]
idθ_EFE = [-logpdf(pθ_EFE[k], sys_coefficients) for k in 1:N]

plot(xlabel="time (steps)", ylabel="-logpdf θ*", size=(900,400))
plot!(idθ_MSE, label="MSE")
plot!(idθ_EFE, label="EFE")
```

```julia
CC_MSE = round(sum(abs.(u_MSE)), digits=2)
CC_EFE = round(sum(abs.(u_EFE)), digits=2)

plot(xlabel="time (steps)", ylabel="control", size=(900,400))
plot!(u_MSE, label="Total cost for MSE = $CC_MSE")
plot!(u_EFE, label="Total cost for EFE = $CC_EFE")
```

```julia

```
