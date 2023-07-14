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
    display_name: Julia 1.8.3
    language: julia
    name: julia-1.8
---

# Bayesian Black-box model-predictive control

In this notebook, I test the NARX-EFE agent on a single pendulum setting.


## System: single pendulum

Consider a single pendulum with angle $\theta$ evolving according to:

$$ \ddot{\theta} = - \frac{\mathcal{g}}{l} sin(\theta) \, .$$

where $l$ is length and $\mathcal{g}$ is Earth's gravity.

```julia
using Revise
using Optim
using RxInfer
using LinearAlgebra
using ProgressMeter
using Distributions
using Plots; default(grid=false, label="", linewidth=3,margin=10Plots.pt)
include("../util.jl")
```

```julia
mutable struct Pendulum

    state       ::Vector{Float64}
    sensor      ::Float64
    
    mass        ::Float64
    length      ::Float64
    mnoise_sd   ::Float64 # measurement noise standard deviation

    function Pendulum(init_state, mass, length, mnoise_sd)
        
        init_sensor = init_state[1] + mnoise_sd*randn()
        
        return new(init_state, init_sensor, mass, length, mnoise_sd)
    end
end
```

```julia
function params(sys::Pendulum)
    return (sys.mass, sys.length)
end
```

```julia
function dzdt(state::Vector, u::Float64, params::Tuple)
    
    mass, length = params
    
    return [state[2]; -9.81/length * sin(state[1]) + 1/mass * u]    
end
```

```julia
function RK4(sys::Pendulum, u::Float64; Δt::Float64=1.0)
    
    K1 = dzdt(sys.state          , u, params(sys))
    K2 = dzdt(sys.state + K1*Δt/2, u, params(sys))
    K3 = dzdt(sys.state + K2*Δt/2, u, params(sys))
    K4 = dzdt(sys.state + K3*Δt  , u, params(sys))
    
    return Δt/6 * (K1 + 2K2 + 2K3 + K4)
end
```

```julia
function update!(sys::Pendulum, u::Float64; Δt::Float64 = 1.0)
    
    sys.state  = sys.state + RK4(sys, u, Δt=Δt)
    
    sys.sensor = sys.state[1] + sys.mnoise_sd * randn()
    
end
```

```julia
# Define system
pendulum_mass = 2.0
pendulum_length = 0.8
mnoise_sd = 1e-2

init_state = [π/2, 0.0]

pendulum = Pendulum(init_state, pendulum_mass, pendulum_length, mnoise_sd)
```

```julia
T = 100
Δt = 0.05
time = range(0.0, step=Δt, length=T)

states = zeros(2,T)
observations = zeros(T)

controls = zeros(T)

for k in 1:T
    
    states[:,k] = pendulum.state
    observations[k] = pendulum.sensor
    
    update!(pendulum, controls[k], Δt=Δt)
    
end

plot(xlabel="time [s]",ylabel="angle", size=(900,300))
plot!(time, states[1,:], color="blue", label="state")
scatter!(time, observations, color="black", label="measurements")
```

## NARX parameter estimation

```julia
# Length of trial
T      = 1200
time   = range(0.0, step=Δt, length=T);
thorizon = 5;
```

```julia
# Degree
H = 5

# Basis expansion
# ϕ(x; N::Int64 = 10) = [1; cat([[cos.(2π*n*x); sin.(2π*n*x)] for n = 1:N]..., dims=1)]
ϕ(x; D::Integer = 1) = cat([1; [x.^d for d in 1:D]]...,dims=1)
# ϕ(x; N::Integer = 1) = [1; x; x.^2]
# ϕ(x; N::Integer = 1) = [1; x]
```

```julia
# Delay order
Lx = 3
Lu = 3

# Model order
M = size(ϕ(zeros(Lx+Lu), D=H),1)
```

```julia
# Control policy
# Ω = 10 .^range(-6,stop=2, length=30)
# u_ = 100*mean([sin.(ω.*time) for ω in Ω]);

# function u(t)
#     u = 1.0
#     if t > 1.0
#         u = 0.0
#     end
#     return
# end
# u_ = u(range(0.0, step=nt*dt, length=T+thorizon))

# Pulse
# u_ = [zeros(10); ones(100); zeros(T-110)];
A  = rand(10)*200 .- 100
Ω  = rand(10)*3
u_ = mean([A[i]*sin.(Ω[i].*time) for i = 1:10])

plot(time, u_, size=(900,300), color="red", xlabel="time (sec)", ylabel="control")
```

```julia
@model function NARX(pθ_k, pτ_k)
    
    ϕ = datavar(Vector{Float64})
    y = datavar(Float64)
    
    # Priors
    θ  ~ MvNormalMeanCovariance(mean(pθ_k), cov(pθ_k))
    τ  ~ GammaShapeRate(shape(pτ_k), rate(pτ_k))
        
    # Likelihood
    y ~ NormalMeanPrecision(dot(θ,ϕ), τ)
end
```

```julia
function future(controls, xbuffer, ubuffer, params; time_horizon=1)
    
    μ_y = zeros(time_horizon)
    σ_y = zeros(time_horizon)

    # Unpack parameters
    mθ, mτ = params

    # Recursive buffer
    vbuffer = 1e-8*ones(length(mθ))
    
    for t in 1:time_horizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        
        # Prediction
        μ_y[t] = dot(mθ, ϕ([xbuffer; ubuffer], D=H))
        σ_y[t] = sqrt(mθ'*diagm(vbuffer)*mθ + inv(mτ))
        
        # Update previous 
        xbuffer = backshift(xbuffer, μ_y[t])
        vbuffer = backshift(vbuffer, σ_y[t]^2)     
        
    end
    return μ_y, σ_y
end
```

```julia
# Specify prior distributions
pτ0 = GammaShapeRate(1e0, 1e0)
pθ0 = MvNormalMeanPrecision(zeros(M), 1e-4diagm(ones(M)))
```

```julia
# Initialize beliefs
pτ = [pτ0]
pθ = [pθ0]
```

```julia
# Clear memory
pτ = [pτ[end]]
pθ = [pθ[end]]
```

```julia
# VI options
num_iters = 5;
constraints = @constraints begin 
    q(θ, τ) = q(θ)q(τ)
end
```

```julia
# Preallocate
z_ = zeros(2,T)
y_ = zeros(T)
u_ = zeros(T)
pred_m = zeros(thorizon,T)
pred_s = zeros(thorizon,T)

for j in 1:30

    A  = randn(10)*5
    Ω  = rand(10)*3
    u_ = mean([A[i]*sin.(Ω[i].*time) for i = 1:10])

    # Initial state
    init_state = [randn(), 0.0]

    # Start system
    pendulum = Pendulum(init_state, pendulum_mass, pendulum_length, mnoise_sd)

    # Preallocate
    z_ = zeros(2,T)
    y_ = zeros(T)
    ϕ_ = zeros(M,T)
    xbuffer = zeros(Lx)
    ubuffer = zeros(Lu)
    pred_m = zeros(thorizon,T)
    pred_s = zeros(thorizon,T)

    @showprogress for (k,t) in enumerate(time[1:T-thorizon])

        "Act upon environment"

        # Alter system state based on control
        update!(pendulum, u_[k], Δt=Δt)

        "Update parameter beliefs"

        ubuffer = backshift(ubuffer, u_[k])
        ϕ_[:,k] = ϕ([xbuffer; ubuffer], D=H)

        # VI iterations
        results = inference(
            model         = NARX(pθ[end], pτ[end]), 
            data          = (y = pendulum.sensor, ϕ = ϕ_[:,k]), 
            initmarginals = (θ = pθ0, τ = pτ0),
            initmessages  = (θ = pθ0, τ = pτ0),
            returnvars    = (θ = KeepLast(), τ = KeepLast(),),
            constraints   = constraints, 
            iterations    = num_iters,
        )

        # Update beliefs
        push!(pθ, results.posteriors[:θ])
        push!(pτ, results.posteriors[:τ])

        xbuffer = backshift(xbuffer, pendulum.sensor)

        # Store future predictions
        pred_m[:,k], pred_s[:,k] = future(u_[k:k+thorizon], 
                                          xbuffer, 
                                          ubuffer, 
                                          (mean(pθ[end]), mode(pτ[end])), 
                                          time_horizon=thorizon)

        # Track states and sensor measurements
        z_[:,k] = pendulum.state
        y_[k]   = pendulum.sensor

    end
end
```

```julia
p1 = plot(time, z_[1,:], color="blue", label="state", ylabel="angle")
plot!(time, pred_m[1,:], ribbon=pred_s[1,:], color="purple", label="prediction")
scatter!( time, y_, color="black", markersize=2, label="measurement")
p4 = plot(time, u_, color="red", ylabel="controls", xlabel="time [s]")

plot(p1,p4, layout=grid(2,1, heights=[.7, .3]), size=(900,600))
```

```julia
savefig("figures/NARX-EFE-pendulum_filtering.png")
```

```julia
scatter(mean(pθ[end]))
```

```julia
ix = 300:400
p2 = plot(size=(900,300), xlabel="time [s]", ylabel="angle")
plot!(time[ix], z_[1,ix], color="blue", label="state")
# plot!(time[ix], pred_m[end,ix .- thorizon], ribbon=pred_s[end,ix .- thorizon], color="purple", label="prediction")
plot!(time[ix], pred_m[end,ix .- thorizon], color="purple", label="prediction")
scatter!(time[ix], y_[ix], color="black", label="measurements", title="final horizon predictions")
```

```julia
limsb = [minimum(z_[2,:])*1.1, maximum(z_[2,:])*1.1]

window = 20

anim = @animate for k in 2:2:(T-thorizon-1)
    
    if k <= window
        plot(time[1:k], z_[1,1:k], color="blue", xlims=(time[1], time[window+thorizon+1]+0.5), label="past data", ylabel="state box", xlabel="time (sec)", ylims=limsb, size=(900,300))
        plot!(time[k:k+thorizon], z_[1,k:k+thorizon], color="purple", label="true future", linestyle=:dot)
        plot!(time[k+1:k+thorizon], pred_m[:,k], ribbon=pred_s[:,k], label="predicted future", color="orange", legend=:topleft)
    else
        plot(time[k-window:k], z_[1,k-window:k], color="blue", xlims=(time[k-window], time[k+thorizon+1]+0.5), label="past data", ylabel="state box", xlabel="time (sec)", ylims=limsb, size=(900,300))
        plot!(time[k:k+thorizon], z_[1,k:k+thorizon], color="purple", label="true future", linestyle=:dot)
        plot!(time[k+1:k+thorizon], pred_m[:,k], ribbon=pred_s[:,k], label="prediction", color="orange", legend=:topleft)
    end
end
gif(anim, "figures/NARX-EFE-pendulum_prediction.gif", fps=24)
```

## Expected Free Energy minimization

```julia
function EFE(control, xbuffer, ubuffer, goalp, params; λ=0.01, time_horizon=1)
    "Expected Free Energy"
    
    # Unpack goal state
    μ_star, σ_star = goalp

    # Unpack parameters
    mθ, mτ = params

    # Recursive buffer
    vbuffer = 1e-8*ones(length(mθ))
    
    cEFE = 0
    for t in 1:time_horizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, control[t])
        
        # Prediction
        μ_y = dot(mθ, ϕ([xbuffer; ubuffer], D=H))
        σ_y = sqrt(mθ'*diagm(vbuffer)*mθ + inv(mτ))

        # Calculate conditional entropy
        ambiguity = 0.5(log(2pi) + log(σ_y))
        
        # Risk as KL between marginal and goal prior
        risk = 0.5*(log(σ_star/σ_y) + (μ_y - μ_star)'*inv(σ_star)*(μ_y - μ_star) + tr(inv(σ_star)*σ_y))
        
        # Add to cumulative EFE
        cEFE += risk + ambiguity# + λ*control[t]^2
        
        # Update previous 
        xbuffer = backshift(xbuffer, μ_y)
        vbuffer = backshift(vbuffer, σ_y^2)        
    end
    return cEFE
end;
```

```julia
# Length of trial
T      = 500
time   = range(0.0, step=Δt, length=T)
thorizon = 5;

# VI options
num_iters = 5;
constraints = @constraints begin 
    q(θ, τ) = q(θ)q(τ)
end

# Set control properties
setpoint = 3.141592
goal_pdf = (setpoint, 1e-6)
u_lims = (-100, 100)

# Initial state
init_state = [randn(), 0.0]

# Start system
pendulum = Pendulum(init_state, pendulum_mass, pendulum_length, mnoise_sd)

# Preallocate
z_ = zeros(2,T)
y_ = zeros(T)
u_ = zeros(T+1)
ϕ_k = zeros(M)
xbuffer = zeros(Lx)
ubuffer = zeros(Lu)
pred_m = zeros(thorizon,T)
pred_s = zeros(thorizon,T)
fe = zeros(num_iters, T)

@showprogress for (k,t) in enumerate(time)
    
    "Act upon environment"
    
    update!(pendulum, u_[k], Δt=Δt)
    
    "Update parameter beliefs"
    
    # Inputs
    ubuffer = backshift(ubuffer, u_[k])
    ϕ_k = ϕ([xbuffer; ubuffer], D=H)
    
    # VI iterations
    results = inference(
        model         = NARX(pθ[end], pτ[end]), 
        data          = (y = pendulum.sensor, ϕ = ϕ_k),
        constraints   = constraints, 
        iterations    = num_iters, 
        initmarginals = (θ = pθ0, τ = pτ0),
        initmessages  = (θ = pθ0, τ = pτ0),
        returnvars    = (θ = KeepLast(), τ = KeepLast(),),        
        free_energy   = true,
    )
    fe[:,k] = results.free_energy
    
    # Update beliefs
    push!(pθ, results.posteriors[:θ])
    push!(pτ, results.posteriors[:τ])
    
    "Optimal control"
    
    # Extract MAP parameters
    mθ = mode(results.posteriors[:θ])
    mτ = mode(results.posteriors[:τ])
    
    # Objective function
    J(policy) = EFE(policy, xbuffer, ubuffer, goal_pdf, (mθ, mτ), λ=1e-5, time_horizon=thorizon)

    # Minimize
    results = optimize(J, u_lims[1], u_lims[2], zeros(thorizon), Fminbox(LBFGS()), autodiff=:forward)
    
    # Control law
    policy = Optim.minimizer(results)
    u_[k+1] = policy[1]
    
    "Bookkeeping"
    
    # Store future predictions
    pred_m[:,k], pred_s[:,k] = future(policy, 
                                      xbuffer, 
                                      ubuffer, 
                                      (mean(pθ[end]), mode(pτ[end])), 
                                      time_horizon=thorizon)
    
    # Update previous observations buffer
    xbuffer = backshift(xbuffer, pendulum.sensor)

    # Track states and sensors
    z_[:,k] = pendulum.state
    y_[k]   = pendulum.sensor
    
end
```

```julia
plot(fe[end,:] .- fe[1,:])
```

```julia
p1 = plot(time, z_[1,:], color="blue", label="state", ylabel="angle")
scatter!( time, y_, color="black", markersize=2, label="measurement")
hline!([setpoint], color="green", ylims=[0., 6.])
p4 = plot(time, u_[1:end-1], color="red", ylabel="controls", xlabel="time [s]")

plot(p1,p4, layout=grid(2,1, heights=[.7, .3]), size=(900,400))
```

```julia
savefig("figures/NARX-EFE-pendulum_trial00.png")
```

```julia
limsb = [minimum(z_[2,:])*1.5, maximum(z_[2,:])*1.5]

window = 20

anim = @animate for k in 2:2:(T-thorizon-1)
    
    if k <= window
        plot(time[1:k], z_[1,1:k], color="blue", xlims=(time[1], time[window+thorizon+1]+0.5), label="past data", ylabel="state box", xlabel="time (sec)", ylims=limsb, size=(900,300))
        plot!(time[k:k+thorizon], z_[1,k:k+thorizon], color="purple", label="true future", linestyle=:dot)
        plot!(time[k+1:k+thorizon], pred_m[:,k], ribbon=pred_s[:,k], label="predicted future", color="orange", legend=:topleft)
        hline!([setpoint], color="green", ylims=[0., 6.])
    else
        plot(time[k-window:k], z_[1,k-window:k], color="blue", xlims=(time[k-window], time[k+thorizon+1]+0.5), label="past data", ylabel="state box", xlabel="time (sec)", ylims=limsb, size=(900,300))
        plot!(time[k:k+thorizon], z_[1,k:k+thorizon], color="purple", label="true future", linestyle=:dot)
        plot!(time[k+1:k+thorizon], pred_m[:,k], ribbon=pred_s[:,k], label="prediction", color="orange", legend=:topleft)
        hline!([setpoint], color="green", ylims=[0., 6.])
    end
end
gif(anim, "figures/NARX-EFE-pendulum_plan_trial00.gif", fps=24)
```

```julia

```

```julia

```
