using RxInfer
using LinearAlgebra


@model function NARX(pθ_k, pτ_k)
    
    ϕ = datavar(Vector{Float64})
    y = datavar(Float64)
    
    # Priors
    θ  ~ MvNormalMeanCovariance(mean(pθ_k), cov(pθ_k))
    τ  ~ GammaShapeRate(shape(pτ_k), rate(pτ_k))
        
    # Likelihood
    y ~ NormalMeanPrecision(dot(θ,ϕ), τ)
end

ϕ(x; D::Integer = 1) = cat([1; [x.^d for d in 1:D]]...,dims=1)

function predictions(policy, xbuffer, ubuffer, params; time_horizon=1)
    
    μ_y = zeros(time_horizon)
    σ_y = zeros(time_horizon)

    # Unpack parameters
    mθ, mτ = params

    # Recursive buffer
    vbuffer = 1e-8*ones(length(mθ))
    
    for t in 1:time_horizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, policy[t])
        
        # Prediction
        μ_y[t] = dot(mθ, ϕ([xbuffer; ubuffer], D=H))
        σ_y[t] = sqrt(mθ'*diagm(vbuffer)*mθ + inv(mτ))
        
        # Update previous 
        xbuffer = backshift(xbuffer, μ_y[t])
        vbuffer = backshift(vbuffer, σ_y[t]^2)     
        
    end
    return μ_y, σ_y
end

function EFE(control, xbuffer, ubuffer, goalp, params; λ=0.01, time_horizon=1)
    "Expected Free Energy"
    
    # Unpack goal state
    μ_star, σ_star = goalp

    # Unpack parameters
    mθ, mτ = params

    # Recursive buffer
    vbuffer = 1e-8*ones(length(mθ))
    
    J = 0
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
        J += risk + ambiguity + λ*control[t]^2
        
        # Update previous 
        xbuffer = backshift(xbuffer, μ_y)
        vbuffer = backshift(vbuffer, σ_y^2)        
    end
    return J
end
