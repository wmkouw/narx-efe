module MARXAgents

using Optim
using Distributions
using SpecialFunctions
using LinearAlgebra

export MARXAgent, update!, predictions, crossentropy, mutualinfo, minimizeEFE, minimizeMSE, backshift, update_goals!


mutable struct MARXAgent
    """
    Active inference agent based on a Multivariate Auto-Regressive eXogenous model.

    Parameters are inferred through Bayesian filtering and controls through minimizing expected free energy.
    """

    Dy              ::Integer
    Dx              ::Integer
    Du              ::Integer
    ybuffer         ::Matrix{Float64}
    ubuffer         ::Matrix{Float64}
    delay_inp       ::Integer
    delay_out       ::Integer

    M               ::Matrix{Float64}   # Coefficients mean matrix
    U               ::Matrix{Float64}   # Coefficients row-covariance
    V               ::Matrix{Float64}   # Precision scale matrix
    ν               ::Float64           # Precision degrees-of-freedom
    λ               ::Float64           # Control prior precision

    goals           ::Distribution{Multivariate, Continuous}
    thorizon        ::Integer
    num_iters       ::Integer

    free_energy     ::Float64

    function MARXAgent(coefficients_mean_matrix,
                       coefficients_row_covariance,
                       precision_scale,
                       precision_degrees,
                       goal_prior;
                       Dy::Integer=2,
                       Du::Integer=2,
                       delay_inp::Integer=1, 
                       delay_out::Integer=1, 
                       time_horizon::Integer=1,
                       num_iters::Integer=10,
                       control_prior_precision::Float64=0.0)

        ybuffer = zeros(Dy,delay_out)
        ubuffer = zeros(Du,delay_inp)
        Dx = Du*delay_inp + Dy*delay_out

        free_energy = Inf

        return new(Dy,
                   Dx,
                   Du,
                   ybuffer,
                   ubuffer,
                   delay_inp,
                   delay_out,
                   coefficients_mean_matrix,
                   coefficients_row_covariance,
                   precision_scale,
                   precision_degrees,
                   control_prior_precision,
                   goal_prior,
                   time_horizon,
                   num_iters,
                   free_energy)
    end
end

function update!(agent::MARXAgent, y_k::Vector, u_k::Vector)

    # Short-hand
    M0 = agent.M
    U0 = agent.U
    V0 = agent.V
    ν0 = agent.ν

    # Update input buffer
    agent.ubuffer = backshift(agent.ubuffer, u_k)
    x_k = [agent.ubuffer[:]; agent.ybuffer[:]]

    # Auxiliary variables
    Ω0 = inv(U0)
    xx = x_k*x_k'
    yxMU = (y_k*x_k' + M0'*Ω0)

    # Update parameters
    agent.M = inv(Ω0 + xx)*yxMU'
    agent.U = inv(Ω0 + xx)
    agent.V = inv(inv(V0) + yxMU*inv(Ω0 + xx)*yxMU' + y_k*y_k' + M0'*Ω0*M0)
    agent.ν = ν0 + 1

    # Update output buffer
    agent.ybuffer = backshift(agent.ybuffer, y_k)

    # Update performance metric
    # agent.free_energy = !TODO
end

function params(agent::MARXAgent)
    return agent.M, agent.U, agent.V, agent.ν
end

function marginal_likelihood(agent::MARXAgent, prior_params)
    error("TODO")
end

function posterior_predictive(agent::MARXAgent, x_t)
    "Posterior predictive distribution is matrix T-distributed."

    # ν_t = 2*agent.α
    # m_t = dot(agent.μ, ϕ_t)
    # s2_t = agent.β/agent.α*(1 + ϕ_t'*inv(agent.Λ)*ϕ_t)

    return ν_t, m_t, s2_t
end

function predictions(agent::MARXAgent, controls; time_horizon=1)
    
    m_y = zeros(time_horizon)
    v_y = zeros(time_horizon)

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer
    
    for t in 1:time_horizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        ϕ_t = pol([ybuffer; ubuffer], degree=agent.pol_degree)

        ν_t, m_t, s2_t = posterior_predictive(agent, ϕ_t)
        
        # Prediction
        m_y[t] = m_t
        v_y[t] = s2_t * ν_t/(ν_t - 2)
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y[t])
        
    end
    return m_y, v_y
end

function mutualinfo(agent::MARXAgent, x)
    "Mutual information between parameters and posterior predictive (constant terms dropped)"
    return 
end

function crossentropy(agent::MARXAgent, goal::Distribution{Multivariate, Continuous}, m_pred, v_pred)
    "Cross-entropy between posterior predictive and goal prior (constant terms dropped)"  
    return ( v_pred + (m_pred - mean(goal))^2 ) / ( 2var(goal) )
end 

function EFE(agent::MARXAgent, goals, controls)
    "Expected Free Energy"

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer
    
    J = 0
    for t in 1:agent.thorizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        x_t = [ubuffer; ybuffer]

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

function minimizeEFE(agent::MARXAgent, goals; u_0=nothing, time_limit=10, verbose=false, control_lims::Tuple=(-Inf,Inf))
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

function backshift(x::AbstractMatrix, a::Vector)
    "Shift elements rightwards and add element"
    return [a x[:,1:end-1]]
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

function update_goals!(x::AbstractVector, g::Distribution{Univariate, Continuous})
    "Move goals forward and add a final goal"
    circshift!(x,-1)
    x[end] = g
end

end
