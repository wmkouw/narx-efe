using RxInfer
using LinearAlgebra


ϕ(x; D::Integer = 1) = cat([1; [x.^d for d in 1:D]]...,dims=1)

@model function NARX(pθ_k, pτ_k)
    
    ϕ = datavar(Vector{Float64})
    y = datavar(Float64)
    
    # Priors
    θ  ~ MvNormalMeanCovariance(mean(pθ_k), cov(pθ_k))
    τ  ~ GammaShapeRate(shape(pτ_k), rate(pτ_k))
        
    # Likelihood
    y ~ NormalMeanPrecision(dot(θ,ϕ), τ)
end

function predictions(policy, xbuffer, ubuffer, params; time_horizon=1)
    
    m_y = zeros(time_horizon)
    v_y = zeros(time_horizon)

    # Unpack parameters
    mθ,Sθ,mτ = params

    # Recursive buffer
    vbuffer = 1e-8*ones(length(mθ))
    
    for t in 1:time_horizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        
        # Prediction
        ϕ_k = ϕ([xbuffer; ubuffer], D=H)
        m_y[t] = dot(mθ, ϕ_k)
        v_y[t] = ϕ_k'*Sθ*ϕ_k + inv(mτ)
        
        # Update previous 
        xbuffer = backshift(xbuffer, m_y[t])
        vbuffer = backshift(vbuffer, v_y[t])     
        
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
end;
