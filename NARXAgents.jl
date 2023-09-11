module NARXAgents

using Optim
using RxInfer
using Distributions
using SpecialFunctions
using LinearAlgebra

export NARXAgent, update!, predictions, pol, crossentropy, mutualinfo, minimizeEFE, minimizeMSE, backshift, update_goals!


mutable struct NARXAgent
    """
    Active inference agent based on a Nonlinear Auto-Regressive eXogenous model.

    Parameters are inferred through Bayesian filtering and controls through minimizing expected free energy.
    """

    ybuffer         ::Vector{Float64}
    ubuffer         ::Vector{Float64}
    delay_inp       ::Integer
    delay_out       ::Integer
    pol_degree      ::Integer
    order           ::Integer

    μ               ::Vector{Float64}   # Coefficients mean
    Λ               ::Matrix{Float64}   # Coefficients precision
    α               ::Float64           # Likelihood precision shape
    β               ::Float64           # Likelihood precision rate
    λ               ::Float64           # Control prior precision

    goals           ::Union{NormalDistributionsFamily, Vector}
    thorizon        ::Integer
    num_iters       ::Integer

    free_energy     ::Float64

    function NARXAgent(coefficients_mean,
                       coefficients_precision,
                       noise_shape,
                       noise_rate; 
                       goal_prior=NormalMeanVariance(0.0, 1.0),
                       delay_inp::Integer=1, 
                       delay_out::Integer=1, 
                       pol_degree::Integer=1,
                       thorizon::Integer=1,
                       num_iters::Integer=10,
                       control_prior_precision::Float64=0.0)

        ybuffer = zeros(delay_out)
        ubuffer = zeros(delay_inp+1)

        order = size(pol(zeros(1 + delay_inp + delay_out), degree=pol_degree),1)
        if order != length(coefficients_mean) 
            error("Dimensionality of coefficients and model order do not match.")
        end

        free_energy = Inf

        return new(ybuffer,
                   ubuffer,
                   delay_inp,
                   delay_out,
                   pol_degree,
                   order,
                   coefficients_mean,
                   coefficients_precision,
                   noise_shape,
                   noise_rate,
                   control_prior_precision,
                   goal_prior,
                   thorizon,
                   num_iters,
                   free_energy)
    end
end

pol(x; degree::Integer = 1) = cat([1.0; [x.^d for d in 1:degree]]...,dims=1)

function update!(agent::NARXAgent, y::Float64, u::Float64)

    agent.ubuffer = backshift(agent.ubuffer, u)
    ϕ = pol([agent.ybuffer; agent.ubuffer], degree=agent.pol_degree)

    μ0 = agent.μ
    Λ0 = agent.Λ
    α0 = agent.α
    β0 = agent.β

    agent.μ = inv(ϕ*ϕ' + Λ0)*(ϕ*y + Λ0*μ0)
    agent.Λ = ϕ*ϕ' + Λ0
    agent.α = α0 + 1/2
    agent.β = β0 + 1/2*(y^2 + μ0'*Λ0*μ0 - (ϕ*y + Λ0*μ0)'*inv(ϕ*ϕ' + Λ0)*(ϕ*y + Λ0*μ0))

    agent.ybuffer = backshift(agent.ybuffer, y)

    agent.free_energy = -log(marginal_likelihood(agent, (μ0, Λ0, α0, β0)))
end

function params(agent::NARXAgent)
    return agent.μ, agent.Λ, agent.α, agent.β
end

function marginal_likelihood(agent::NARXAgent, prior_params)

    μn, Λn, αn, βn = params(agent)
    μ0, Λ0, α0, β0 = prior_params

    return (det(Λn)^(-1/2)*gamma(αn)*βn^αn)/(det(Λ0)^(-1/2)*gamma(α0)*β0^α0) * (2π)^(-1/2)
end

function posterior_predictive(agent::NARXAgent, ϕ)
    "Posterior predictive distribution is location-scale t-distributed"

    ν_star = 2*agent.α
    μ_star = dot(agent.μ, ϕ)
    σ_star = sqrt( agent.β/agent.α*(1 + ϕ'*inv(agent.Λ)*ϕ) ) 

    return ν_star, μ_star, σ_star
end

function predictions(agent::NARXAgent, controls; time_horizon=1)
    
    m_y = zeros(time_horizon)
    v_y = zeros(time_horizon)

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer
    
    for t in 1:time_horizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        ϕ_t = pol([ybuffer; ubuffer], degree=agent.pol_degree)

        ν_t, μ_t, σ_t = posterior_predictive(agent, ϕ_t)
        
        # Prediction
        m_y[t] = μ_t
        v_y[t] = σ_t^2 * ν_t/(ν_t - 2)
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y[t])
        
    end
    return m_y, v_y
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

    ν = 2agent.α
    return ( (v_pred).*ν/(ν-2) + (m_pred - mean(goal))^2)/(2var(goal))
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
        ϕ_t = pol([ybuffer; ubuffer], degree=agent.pol_degree)
        
        # Prediction
        m_y = dot(agent.μ, ϕ_t)
        v_y = (ϕ_k'*Σ*ϕ_k + β/α)*2α/(2α-2)
        
        # Accumulate EFE
        J += mutualinfo(agent, ϕ_k) + crossentropy(agent, goals[t], m_y,v_y) + agent.λ*controls[t]^2
        
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
