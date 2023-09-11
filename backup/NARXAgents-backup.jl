module NARXAgents

using Optim
using RxInfer
using Distributions
using SpecialFunctions
using LinearAlgebra

export NARXAgent, update!, predictions, pol, crossentropy, mutualinfo, minimizeEFE, minimizeMSE, backshift, update_goals!


mutable struct NARXAgent
    """
    Active inference agent based on a Nonlinear Auto-Regressive model with eXogenous control.

    Parameters are inferred through Bayesian filtering and controls through minimizing expected free energy.
    """

    model           ::FactorGraphModel
    constraints     ::ConstraintsSpecification

    free_energy     ::Vector{Float64}
    qθ              ::NormalDistributionsFamily
    qτ              ::GammaDistributionsFamily 
    goals           ::Union{NormalDistributionsFamily, Vector}

    thorizon        ::Integer
    num_iters       ::Integer
    control_prior   ::Float64

    delay_inp       ::Integer
    delay_out       ::Integer
    pol_degree      ::Integer
    order           ::Integer

    ybuffer         ::Vector{Float64}
    ubuffer         ::Vector{Float64}

    function NARXAgent(prior_coefficients, 
                       prior_precision; 
                       goal_prior=NormalMeanVariance(0.0, 1.0),
                       delay_inp::Integer=1, 
                       delay_out::Integer=1, 
                       pol_degree::Integer=1,
                       thorizon::Integer=1,
                       num_iters::Integer=10,
                       control_prior::Float64=0.0)

        model, _ = create_model(NARX(prior_coefficients, prior_precision))

        constraints = @constraints begin 
            q(θ, τ) = q(θ)q(τ)
        end

        free_energy = [Inf]
        ybuffer = zeros(delay_out)
        ubuffer = zeros(delay_inp+1)

        order = size(pol(zeros(1 + delay_inp + delay_out), degree=pol_degree),1)
        if order != length(prior_coefficients) 
            error("Dimensionality of coefficients prior and model order do not match.")
        end

        return new(model, 
                   constraints,
                   free_energy,
                   prior_coefficients, 
                   prior_precision,
                   goal_prior,
                   thorizon,
                   num_iters,
                   control_prior,
                   delay_inp,
                   delay_out,
                   pol_degree,
                   order,
                   ybuffer,
                   ubuffer)
    end
end

pol(x; degree::Integer = 1) = cat([1.0; [x.^d for d in 1:degree]]...,dims=1)

@model function NARX(pθ, pτ)
    
    ϕ = datavar(Vector{Float64})
    y = datavar(Float64)
    
    # Priors
    θ  ~ MvNormalMeanCovariance(mean(pθ), cov(pθ))
    τ  ~ GammaShapeRate(shape(pτ), rate(pτ))
        
    # Likelihood
    y ~ NormalMeanPrecision(dot(θ,ϕ), τ)

    return θ,τ
end

function update!(agent::NARXAgent, observation::Float64, control::Float64)

    agent.ubuffer = backshift(agent.ubuffer, control)
    input = pol([agent.ybuffer; agent.ubuffer], degree=agent.pol_degree)

    results = inference(
        model         = NARX(agent.qθ, agent.qτ), 
        data          = (y = observation, ϕ = input), 
        initmarginals = (θ = agent.qθ, τ = agent.qτ),
        initmessages  = (θ = agent.qθ, τ = agent.qτ),
        returnvars    = (θ = KeepLast(), τ = KeepLast()),
        constraints   = agent.constraints, 
        iterations    = agent.num_iters,
        free_energy   = true,
    )

    agent.free_energy = results.free_energy
    agent.qθ = results.posteriors[:θ]
    agent.qτ = results.posteriors[:τ]

    agent.ybuffer = backshift(agent.ybuffer, observation)
end

function predictions(agent::NARXAgent, controls; time_horizon=1)
    
    m_y = zeros(time_horizon)
    v_y = zeros(time_horizon)

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer

    # Unpack parameters
    μ = mean( agent.qθ)
    Σ = cov(  agent.qθ)
    α = shape(agent.qτ)
    β = rate( agent.qτ)
    
    for t in 1:time_horizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        ϕ_t = pol([ybuffer; ubuffer], degree=agent.pol_degree)
        
        # Prediction
        m_y[t] = dot(μ, ϕ_t)
        v_y[t] = (ϕ_t'*Σ*ϕ_t + β/α)*2α/(2α-2)
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y[t])
        
    end
    return m_y, v_y
end

function ambiguity(agent::NARXAgent, ϕ_k)
    "Entropies of parameters minus joint entropy of future observation and parameters"
    
    μ = mean( agent.qθ)
    Σ = cov(  agent.qθ)
    α = shape(agent.qτ)
    β = rate( agent.qτ)
    
    # S_k = [Σ                 Σ*ϕ_k;
    #        ϕ_k'*Σ   ϕ_k'*Σ*ϕ_k+β/α]
    
    Dθ = length(μ)
    
    # return logdet(Σ)/2 -logdet(S_k)/2 -(1+(Dθ-2)/2)*log(β) +(1+(Dθ-2)/2)*digamma(α)
    # return -log(β/α) -(1+(Dθ-2)/2)*log(β) +(1+(Dθ-2)/2)*digamma(α)
    return -(logdet(Σ) + logdet(ϕ_k'*Σ*ϕ_k+β/α))/2
end

function mutualinfo(agent::NARXAgent, ϕ_t)
    "Entropies of parameters minus joint entropy of future observation and parameters"
    
    α = shape(agent.qτ)
    β = rate( agent.qτ)
    μ = mean( agent.qθ)
    Σ = cov(  agent.qθ)
    # D = length(μ)
    
    # S0 = [Σ       Σ*ϕ_k;     ϕ_k'*Σ   ϕ_k'*Σ*ϕ_k+β/α]
    # S1 = [Σ  zeros(D,1); zeros(1,D)   ϕ_k'*Σ*ϕ_k+β/α]
    # return 1/2(tr(inv(S1)*S0) +logdet(S1) -logdet(S0))
    return -1/2*log(ϕ_t'*Σ*ϕ_t + β/α)
end

function crossentropy(agent::NARXAgent, goal::NormalMeanVariance, m_pred, v_pred)
    "Entropy of marginal prediction + KL-divergence between marginal prediction and goal prior"  

    α = shape(agent.qτ)
    return ( (v_pred).*2α/(2α-2) + (m_pred - mean(goal))^2)/(2var(goal))
end 

function EFE(agent::NARXAgent, goals, controls)
    "Expected Free Energy"

    μ = mean( agent.qθ)
    Σ = cov(  agent.qθ)
    α = shape(agent.qτ)
    β = rate( agent.qτ)

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer
    
    J = 0
    for t in 1:agent.thorizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        ϕ_k = pol([ybuffer; ubuffer], degree=agent.pol_degree)
        
        # Prediction
        m_y = dot(μ, ϕ_k)
        v_y = (ϕ_k'*Σ*ϕ_k + β/α)*2α/(2α-2)
        
        # Accumulate EFE
        J += mutualinfo(agent, ϕ_k) + crossentropy(agent, goals[t], m_y,v_y) + agent.control_prior*controls[t]^2
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y)        
    end
    return J
end

function MSE(agent::NARXAgent, goals, controls)
    "Mean Squared Error between prediction and setpoint."

    μ = mean( agent.qθ)

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer
    
    J = 0
    for t in 1:agent.thorizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        ϕ_k = pol([ybuffer; ubuffer], degree=agent.pol_degree)
        
        # Prediction
        m_y = dot(μ, ϕ_k)
        
        # Accumulate objective function
        J += (mean(goals[t]) - m_y)^2 + agent.control_prior*controls[t]^2
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y)        
    end
    return J
end

function minimizeEFE(agent::NARXAgent, goals; u_0=nothing, time_limit=10, verbose=false, control_lims::Tuple=(-Inf,Inf))
    "Minimize EFE objective and return policy."

    if isnothing(u_0); u_0 = 1e-8*randn(agent.thorizon); end
    opts = Optim.Options(time_limit=time_limit, 
                         show_trace=verbose, 
                         allow_f_increases=true, 
                         g_tol=1e-12, 
                         show_every=10,
                         iterations=10_000)

    # Objective function
    J(u) = EFE(agent, goals, u)

    # Constrained minimization procedure
    results = optimize(J, control_lims..., u_0, Fminbox(LBFGS()), opts, autodiff=:forward)

    return Optim.minimizer(results)
end

function minimizeMSE(agent::NARXAgent, goals; u_0=nothing, time_limit=10, verbose=false, control_lims::Tuple=(-Inf,Inf))
    "Minimize MSE objective and return policy."

    if isnothing(u_0); u_0 = 1e-8*randn(agent.thorizon); end
    opts = Optim.Options(time_limit=time_limit, 
                         show_trace=verbose, 
                         allow_f_increases=true, 
                         g_tol=1e-12, 
                         show_every=10,
                         iterations=10_000)

    # Objective function
    J(u) = MSE(agent, goals, u)

    # Constrained minimization procedure
    results = optimize(J, control_lims..., u_0, Fminbox(LBFGS()), opts, autodiff=:forward)

    return Optim.minimizer(results)
end

function backshift(x::AbstractVector, a::Number)
    "Shift elements down and add element"

    N = size(x,1)

    # Shift operator
    S = Tridiagonal(ones(N-1), zeros(N), zeros(N-1))

    # Basis vector
    e = [1.0; zeros(N-1)]

    return S*x + e*a
end

function update_goals!(x::AbstractVector, g::NormalMeanVariance)
    "Move goals forward and add a final goal"
    circshift!(x,-1)
    x[end] = g
end

end
