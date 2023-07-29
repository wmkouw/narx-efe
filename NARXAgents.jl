module NARXAgents

using Optim
using RxInfer
using Distributions
using SpecialFunctions
using LinearAlgebra

export NARXAgent, update!, predictions, ϕ, minimizeEFE, minimizeMSE


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

    memory_actions  ::Integer
    memory_senses   ::Integer
    pol_degree      ::Integer
    order           ::Integer

    ybuffer         ::Vector{Float64}
    ubuffer         ::Vector{Float64}

    function NARXAgent(prior_coefficients, 
                       prior_precision; 
                       goal_prior=NormalMeanVariance(0.0, 1.0),
                       memory_actions::Integer=1, 
                       memory_senses::Integer=1, 
                       pol_degree::Integer=1,
                       thorizon::Integer=1,
                       num_iters::Integer=10,
                       control_prior::Float64=1.0)

        model, _ = create_model(NARX(prior_coefficients, prior_precision))

        constraints = @constraints begin 
            q(θ, τ) = q(θ)q(τ)
        end

        free_energy = [Inf]
        ybuffer = zeros(memory_senses)
        ubuffer = zeros(memory_actions)

        order = size(ϕ(zeros(memory_actions+memory_senses), degree=pol_degree),1)
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
                   memory_actions,
                   memory_senses,
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
        ϕ_t = ϕ([ybuffer; ubuffer], degree=agent.pol_degree)
        
        # Prediction
        m_y[t] = dot(μ, ϕ_t)
        v_y[t] = (ϕ_t'*Σ*ϕ_t + 1)*β/α
        
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
    return -log(β/α) -(1+(Dθ-2)/2)*log(β) +(1+(Dθ-2)/2)*digamma(α)
end

function risk(agent::NARXAgent, prediction::Normal)
    "KL-divergence between marginal predicted observation and goal prior"
    
    m_pred, v_pred = mean_var(prediction)
    m_star, v_star = mean_var(agent.goal)
    
    return (log(v_star/v_pred) + (m_pred-m_star)'*inv(v_star)*(m_pred-m_star) + v_pred/v_star)/2
end

function EFE(agent::NARXAgent, controls)
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
        v_y = (ϕ_k'*Σ*ϕ_k + 1)*β/α
        
        # Accumulate EFE
        J += ambiguity(agent, ϕ_k) + risk(agent, Normal(m_y,v_y)) + agent.control_prior*controls[t]^2
        # J += risk(agent, Normal(m_y,v_y)) + agent.control_prior*controls[t]^2
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y)        
    end
    return J
end

function MSE(agent::NARXAgent, controls)
    "Mean Squared Error between prediction and setpoint."

    μ = mean( agent.qθ)

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer

    m_star, _ = mean_var(agent.goal)
    
    J = 0
    for t in 1:agent.thorizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        ϕ_k = ϕ([ybuffer; ubuffer], degree=agent.pol_degree)
        
        # Prediction
        m_y = dot(μ, ϕ_k)
        
        # Accumulate objective function
        J += (m_star - m_y)^2 + agent.control_prior*controls[t]^2
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y)        
    end
    return J
end

function minimizeEFE(agent::NARXAgent; u_0=nothing, time_limit=10, control_lims::Tuple=(-Inf,Inf))
    "Minimize EFE objective and return policy."

    opts = Optim.Options(time_limit=time_limit)
    if isnothing(u_0); u_0 = zeros(agent.thorizon); end

    # Objective function
    J(u) = EFE(agent, u)

    # Constrained minimization procedure
    results = optimize(J, control_lims..., u_0, Fminbox(LBFGS()), opts, autodiff=:forward)

    return Optim.minimizer(results)
end

function minimizeMSE(agent::NARXAgent; u_0=nothing, time_limit=10, control_lims::Tuple=(-Inf,Inf))
    "Minimize MSE objective and return policy."

    opts = Optim.Options(time_limit=time_limit)
    if isnothing(u_0); u_0 = zeros(agent.thorizon); end

    # Objective function
    J(u) = MSE(agent, u)

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

end
