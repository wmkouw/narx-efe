module NARXAgents

using Optim
using RxInfer
using Distributions
using SpecialFunctions
using LinearAlgebra

export NARXAgent, update!, predictions, ϕ, minimizeEFE, minimizeMSE, update_goals!


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
    goal            ::NormalDistributionsFamily

    thorizon        ::Integer
    num_iters       ::Integer
    control_prior   ::Float64

    delay_in        ::Integer
    delay_out       ::Integer
    pol_degree      ::Integer
    order           ::Integer

    ybuffer         ::Vector{Float64}
    ubuffer         ::Vector{Float64}

    function NARXAgent(prior_coefficients, 
                       prior_precision; 
                       goal_prior=NormalMeanVariance(0.0, 1.0),
                       delay_in::Integer=1, 
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
        ubuffer = zeros(1+delay_in)

        order = size(ϕ(zeros(1+delay_in+delay_out), degree=pol_degree),1)
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
                   delay_in,
                   delay_out,
                   pol_degree,
                   order,
                   ybuffer,
                   ubuffer)
    end
end

ϕ(x; degree::Integer = 1) = cat([1; [x.^d for d in 1:degree]]...,dims=1)

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
    memory = ϕ([agent.ybuffer; agent.ubuffer], degree=agent.pol_degree)

    results = inference(
        model         = NARX(agent.qθ, agent.qτ), 
        data          = (y = observation, ϕ = memory), 
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

function predictions(agent::NARXAgent, controls::Vector; time_horizon=1)
    
    p_y = []
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
        ϕ_t = ϕ([ybuffer; ubuffer], degree=agent.pol_degree)
        
        # Prediction
        m_y[t] = dot(μ, ϕ_t)
        # v_y[t] = (ϕ_t'*Σ*ϕ_t + 1)*β/α
        v_y[t] = ϕ_t'*Σ*ϕ_t + β/α

        try
            push!(p_y, Normal(m_y[t], v_y[t]))
        catch
            push!(p_y, NaN)
        end
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y[t])
        
    end
    return p_y
end

function ambiguity(agent::NARXAgent, ϕ_k)
    "Entropies of parameters minus joint entropy of future observation and parameters"
    
    μ = mean( agent.qθ)
    Σ = cov(  agent.qθ)
    α = shape(agent.qτ)
    β = rate( agent.qτ)
    
    # S_k = [Σ                 Σ*ϕ_k;
    #        ϕ_k'*Σ   ϕ_k'*Σ*ϕ_k+β/α]
    
    # Dθ = length(μ)
    
    # return logdet(Σ)/2 -logdet(S_k)/2 -(1+(Dθ-2)/2)*log(β) +(1+(Dθ-2)/2)*digamma(α)
    # return -log(β/α) -(1+(Dθ-2)/2)*log(β) +(1+(Dθ-2)/2)*digamma(α)
    return -1/2*(digamma(α) + log(β))
end

function risk(agent::NARXAgent, prediction::Normal)
    "KL-divergence between marginal predicted observation and goal prior"
    
    m_pred, v_pred = mean_var(prediction)
    m_star, v_star = mean_var(agent.goal)
    
    return (log(v_star/v_pred) + (m_pred-m_star)'*inv(v_star)*(m_pred-m_star) + v_pred/v_star)/2
end

function risk(goal::Union{Bool,NormalMeanVariance}, m_pred, v_pred)
    "KL-divergence between marginal predicted observation and goal prior"

    if goal == false
        return 0.0
    else
        m_star, v_star = mean_var(goal)
        return (log(v_star/v_pred) + (m_pred-m_star)'*inv(v_star)*(m_pred-m_star) + v_pred/v_star)/2
    end
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
        ϕ_k = ϕ([ybuffer; ubuffer], degree=agent.pol_degree)
        
        # Prediction
        m_y = dot(μ, ϕ_k)
        v_y = ϕ_k'*Σ*ϕ_k + β/α
        
        # Accumulate EFE
        J += risk(goals[t], m_y, v_y)
        
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
        ϕ_k = ϕ([ybuffer; ubuffer], degree=agent.pol_degree)
        
        # Prediction
        m_y = dot(μ, ϕ_k)
        
        # Accumulate objective function
        if goal == false
        else
            J += (mean(goals[t]) - m_y)^2
        end
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y)        
    end
    return J
end

function minimizeEFE(agent::NARXAgent, goals; u_0=nothing, time_limit=10, verbose=false, control_lims::Tuple=(-Inf,Inf))
    "Minimize EFE objective and return policy."

    if isnothing(u_0); u_0 = zeros(agent.thorizon); end
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

    if isnothing(u_0); u_0 = zeros(agent.thorizon); end
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
