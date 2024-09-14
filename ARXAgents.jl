module ARXAgents

using Optim
using Distributions
using SpecialFunctions
using LinearAlgebra

export ARXAgent, update!, predictions, crossentropy, mutualinfo, minimizeEFE, minimizeMSE, backshift, update_goals!


mutable struct ARXAgent
    """
    Active inference agent based on an Auto-Regressive eXogenous model.

    Parameters are inferred through Bayesian filtering and controls through minimizing expected free energy.
    """

    ybuffer         ::Vector{Float64}
    ubuffer         ::Vector{Float64}
    delay_inp       ::Integer
    delay_out       ::Integer
    order           ::Integer

    μ               ::Vector{Float64}   # Coefficients mean
    Λ               ::Matrix{Float64}   # Coefficients precision
    α               ::Float64           # Likelihood precision shape
    β               ::Float64           # Likelihood precision rate
    λ               ::Float64           # Control prior precision

    goals           ::Union{Distribution{Univariate, Continuous}, Vector}
    thorizon        ::Integer
    num_iters       ::Integer

    free_energy     ::Float64

    function ARXAgent(coefficients_mean,
                       coefficients_precision,
                       noise_shape,
                       noise_rate; 
                       goal_prior=Normal(0.0, 1.0),
                       delay_inp::Integer=1, 
                       delay_out::Integer=1, 
                       time_horizon::Integer=1,
                       num_iters::Integer=10,
                       control_prior_precision::Float64=0.0)

        ybuffer = zeros(delay_out)
        ubuffer = zeros(delay_inp+1)

        order = zeros(1 + delay_inp + delay_out)
        if order != length(coefficients_mean) 
            error("Dimensionality of coefficients and model order do not match.")
        end

        free_energy = Inf

        return new(ybuffer,
                   ubuffer,
                   delay_inp,
                   delay_out, 
                   order,
                   coefficients_mean,
                   coefficients_precision,
                   noise_shape,
                   noise_rate,
                   control_prior_precision,
                   goal_prior,
                   time_horizon,
                   num_iters,
                   free_energy)
    end
end

function update!(agent::ARXAgent, y::Float64, u::Float64)

    agent.ubuffer = backshift(agent.ubuffer, u)
    x = [agent.ybuffer; agent.ubuffer]

    μ0 = agent.μ
    Λ0 = agent.Λ
    α0 = agent.α
    β0 = agent.β

    agent.μ = inv(x*x' + Λ0)*(x*y + Λ0*μ0)
    agent.Λ = x*x' + Λ0
    agent.α = α0 + 1/2
    agent.β = β0 + 1/2*(y^2 + μ0'*Λ0*μ0 - (x*y + Λ0*μ0)'*inv(x*x' + Λ0)*(x*y + Λ0*μ0))

    agent.ybuffer = backshift(agent.ybuffer, y)

    agent.free_energy = -log(marginal_likelihood(agent, (μ0, Λ0, α0, β0)))
end

function params(agent::ARXAgent)
    return agent.μ, agent.Λ, agent.α, agent.β
end

function marginal_likelihood(agent::ARXAgent, prior_params)

    μn, Λn, αn, βn = params(agent)
    μ0, Λ0, α0, β0 = prior_params

    return (det(Λn)^(-1/2)*gamma(αn)*βn^αn)/(det(Λ0)^(-1/2)*gamma(α0)*β0^α0) * (2π)^(-1/2)
end

function posterior_predictive(agent::ARXAgent, x_t)
    "Posterior predictive distribution is location-scale t-distributed"

    ν_t = 2*agent.α
    m_t = dot(agent.μ, x_t)
    s2_t = agent.β/agent.α*(1 + x_t'*inv(agent.Λ)*x_t)

    return ν_t, m_t, s2_t
end

function predictions(agent::ARXAgent, controls; time_horizon=1)
    
    m_y = zeros(time_horizon)
    v_y = zeros(time_horizon)

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer
    
    for t in 1:time_horizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        x_t = [ybuffer; ubuffer]

        ν_t, m_t, s2_t = posterior_predictive(agent, x_t)
        
        # Prediction
        m_y[t] = m_t
        v_y[t] = s2_t * ν_t/(ν_t - 2)
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y[t])
        
    end
    return m_y, v_y
end

function mutualinfo(agent::ARXAgent, x)
    "Mutual information between parameters and posterior predictive (constant terms dropped)"
    return -1/2*log( agent.β/agent.α*(1 + x'*inv(agent.Λ)*x) )
end

function crossentropy(agent::ARXAgent, goal::Distribution{Univariate, Continuous}, m_pred, v_pred)
    "Cross-entropy between posterior predictive and goal prior (constant terms dropped)"  
    return ( v_pred + (m_pred - mean(goal))^2 ) / ( 2var(goal) )
    # return (m_pred - mean(goal))^2/(2var(goal))
end 

function EFE(agent::ARXAgent, goals, controls)
    "Expected Free Energy"

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer
    
    J = 0
    for t in 1:agent.thorizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        x_t = [ybuffer; ubuffer]

        # Prediction
        ν_t, m_t, s2_t = posterior_predictive(agent, x_t)
        
        m_y = m_t
        v_y = s2_t * ν_t/(ν_t - 2)
        
        # Accumulate EFE
        J += mutualinfo(agent, x_t) + crossentropy(agent, goals[t], m_y, v_y) + agent.λ*controls[t]^2
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y)        
    end
    return J
end

function MSE(agent::ARXAgent, goals, controls)
    "Mean Squared Error between prediction and setpoint."

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer
    
    J = 0
    for t in 1:agent.thorizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        x_t = [ybuffer; ubuffer]
        
        # Prediction
        m_y = dot(agent.μ, x_t)
        
        # Accumulate objective function
        J += (mean(goals[t]) - m_y)^2 + agent.λ*controls[t]^2
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y)        
    end
    return J
end

function minimizeEFE(agent::ARXAgent, goals; u_0=nothing, time_limit=10, verbose=false, control_lims::Tuple=(-Inf,Inf))
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

function minimizeMSE(agent::ARXAgent, goals; u_0=nothing, time_limit=10, verbose=false, control_lims::Tuple=(-Inf,Inf))
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

function backshift(x::AbstractVector, a)
    "Shift elements down and add element"
    circshift!(x,1)
    x[1] = a
    return x
end

function update_goals!(x::AbstractVector, g::Distribution{Univariate, Continuous})
    "Move goals forward and add a final goal"
    circshift!(x,-1)
    x[end] = g
end

function prod!(x::NormalDistributionsFamily, y::NormalDistributionsFamily)
    "Product of Gaussians by moment matching"

    mx,vx = mean_cov(x)
    my,vy = mean_cov(y)

    qmz = mx*my
    qvz = (vx + mx^2)*(vy + my^2) - mx^2*my^2

    return qmz,qvz
end
