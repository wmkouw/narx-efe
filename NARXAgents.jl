module NARXAgents

using RxInfer
using Distributions
using LinearAlgebra

export NARXAgent, update!, ϕ, ambiguity, risk, EFE


mutable struct NARXAgent

    model           ::FactorGraphModel
    constraints     ::ConstraintsSpecification

    qθ              ::NormalDistributionsFamily
    qτ              ::GammaDistributionsFamily 

    num_iters       ::Integer

    memory_actions  ::Integer
    memory_senses   ::Integer
    pol_degree      ::Integer
    order           ::Integer

    ybuffer         ::Vector{Float64}
    ubuffer         ::Vector{Float64}

    function NARXAgent(prior_coefficients, 
                       prior_precision; 
                       memory_actions::Integer=1, 
                       memory_senses::Integer=1, 
                       pol_degree::Integer=1,
                       num_iters::Integer=10)

        model, _ = create_model(NARX(prior_coefficients, prior_precision))

        constraints = @constraints begin 
            q(θ, τ) = q(θ)q(τ)
        end

        ybuffer = zeros(memory_senses)
        ubuffer = zeros(memory_actions)

        order = size(ϕ(zeros(memory_actions+memory_senses), degree=pol_degree),1)
        if order != length(prior_coefficients) 
            error("Dimensionality of coefficients prior and model order do not match.")
        end

        return new(model, 
                   constraints,
                   prior_coefficients, 
                   prior_precision,
                   num_iters,
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
    )

    agent.qθ = results.posteriors[:θ]
    agent.qτ = results.posteriors[:τ]

    agent.ybuffer = backshift(agent.ybuffer, observation)
end

function predictions(controls, xbuffer, ubuffer, params; time_horizon=1)
    
    m_y = zeros(time_horizon)
    v_y = zeros(time_horizon)

    # Unpack parameters
    mθ,Sθ,mτ = params
    
    for t in 1:time_horizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        
        # Prediction
        ϕ_k = ϕ([xbuffer; ubuffer], D=H)
        m_y[t] = dot(mθ, ϕ_k)
        v_y[t] = ϕ_k'*Sθ*ϕ_k + inv(mτ)
        
        # Update previous 
        xbuffer = backshift(xbuffer, m_y[t])
        
    end
    return m_y, v_y
end

function ambiguity(qθ,qτ,ϕ_k)
    "Entropies of parameters minus joint entropy of future observation and parameters"
    
    μ_k, Σ_k = qθ
    α_k, β_k = qτ
    
    S_k = [Σ_k        Σ_k*ϕ_k
           ϕ_k'*Σ_k   ϕ_k'*Σ_k*ϕ_k + β_k/α_k]
    
    Dθ = length(μ_k)
    
    return logdet(Σ_k)/2 -logdet(S_k)/2 -(1+(Dθ-2)/2)*log(β_k) +(1+(Dθ-2)/2)*digamma(α_k)
end

function risk(prediction, goal_prior)
    "KL-divergence between marginal predicted observation and goal prior"
    
    m_pred, v_pred = prediction
    m_star, v_star = goal_prior
    
    return (log(v_star/v_pred) + (m_pred - m_star)'*inv(v_star)*(m_pred - m_star) + tr(inv(v_star)*v_pred))/2
end

function EFE(control, xbuffer, ubuffer, goalp, params; λ=0.01, time_horizon=1)
    "Expected Free Energy"

    # Unpack parameters
    μ_k, Σ_k, α_k, β_k = params
    
    cEFE = 0
    for t in 1:time_horizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, control[t])
        
        # Prediction
        ϕ_k = ϕ([xbuffer; ubuffer], D=H)
        m_y = dot(μ_k, ϕ_k)
        v_y = ϕ_k'*Σ_k*ϕ_k + β_k/α_k
        
        # Add to cumulative EFE
        cEFE += ambiguity((μ_k, Σ_k), (α_k, β_k), ϕ_k) + risk((m_y,v_y),goalp) + λ*control[t]^2
        
        # Update previous 
        xbuffer = backshift(xbuffer, m_y)        
    end
    return cEFE
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
