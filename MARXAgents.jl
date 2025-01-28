module MARXAgents

using Optim
using Distributions
using SpecialFunctions
using LinearAlgebra
using ForwardDiff

export MARXAgent, update!, predictions, posterior_predictive, EFE, crossentropy, mutualinfo, minimizeEFE, update_goals!


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
    Λ               ::Matrix{Float64}   # Coefficients row-covariance
    Ω               ::Matrix{Float64}   # Precision scale matrix
    ν               ::Float64           # Precision degrees-of-freedom
    Υ               ::Matrix{Float64}   # Control prior precision matrix

    goal_prior      ::Distribution{Multivariate, Continuous}
    thorizon        ::Integer
    num_iters       ::Integer

    free_energy     ::Float64

    function MARXAgent(coefficients_mean_matrix,
                       coefficients_row_covariance,
                       precision_scale,
                       precision_degrees,
                       control_prior_precision,
                       goal_prior;
                       Dy::Integer=2,
                       Du::Integer=2,
                       delay_inp::Integer=1, 
                       delay_out::Integer=1, 
                       time_horizon::Integer=1,
                       num_iters::Integer=10)

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
    M = agent.M
    Λ = agent.Λ
    Ω = agent.Ω
    ν = agent.ν

    # Update input buffer
    agent.ubuffer = backshift(agent.ubuffer, u_k)
    x_k = [agent.ubuffer[:]; agent.ybuffer[:]]

    # Auxiliary variables
    X = x_k*x_k'
    Ξ = (x_k*y_k' + Λ*M)

    # Update rules
    agent.ν = ν + 1
    agent.Λ = Λ + X
    agent.Ω = Ω + y_k*y_k' + M'*Λ*M - Ξ'*inv(Λ+X)*Ξ
    agent.M = inv(Λ+X)*Ξ

    # Update output buffer
    agent.ybuffer = backshift(agent.ybuffer, y_k)

    # Update performance metric
    agent.free_energy = -logevidence(agent, y_k, x_k)
end

function params(agent::MARXAgent)
    return agent.M, agent.U, agent.V, agent.ν
end

function logevidence(agent::MARXAgent, y,x)
    η, μ, Ψ = posterior_predictive(agent, x)
    return -1/2*(agent.Dy*log(η*π) -logdet(Ψ) - 2*logmultigamma(agent.Dy, (η+agent.Dy)/2) + 2*logmultigamma(agent.Dy, (η+agent.Dy-1)/2) + (η+agent.Dy)*log(1 + 1/η*(y-μ)'*Ψ*(y-μ)) )
end

function posterior_predictive(agent::MARXAgent, x_t)
    "Posterior predictive distribution is multivariate T-distributed."

    η_t = agent.ν - agent.Dy + 1
    μ_t = agent.M'*x_t
    Ψ_t = (agent.ν-agent.Dy+1)*inv(agent.Ω)*inv(1 + x_t'*inv(agent.Λ)*x_t)

    return η_t, μ_t, Ψ_t
end

function predictions(agent::MARXAgent, controls; time_horizon=1)
    
    m_y = zeros(agent.Dy,time_horizon)
    S_y = zeros(agent.Dy,agent.Dy,time_horizon)

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer
    
    for t in 1:time_horizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[:,t])
        x_t = [ubuffer[:]; ybuffer[:]]

        # Prediction
        η_t, μ_t, Ψ_t = posterior_predictive(agent, x_t)
        m_y[:,t] = μ_t
        S_y[:,:,t] = inv(Ψ_t) * η_t/(η_t - 2)
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y[:,t])
        
    end
    return m_y, S_y
end

function mutualinfo(agent::MARXAgent, x)
    "Mutual information between parameters and posterior predictive (constant terms dropped)"

    _, _, Ψ = posterior_predictive(agent, x)

    return logdet(Ψ)
end

function crossentropy(agent::MARXAgent, x)
    "Cross-entropy between posterior predictive and goal prior (constant terms dropped)"  

    m_star = mean(agent.goal_prior)
    S_star = cov(agent.goal_prior)
    η_t, μ_t, Ψ_t = posterior_predictive(agent, x)

    return 1/2*( η_t/(η_t-2)*tr(inv(S_star)*inv(Ψ_t)) + (μ_t-m_star)'*inv(S_star)*(μ_t-m_star) ) 
end 

function sampleW(agent; num_samples=1)
    "Return samples from a Wishart distribution"

    W = rand(Wishart(agent.ν, inv(agent.Ω)), num_samples)
    A = [rand(MatrixNormal(agent.M, inv(agent.Λ), inv(Wi))) for Wi in W]
    return [(Ai,Wi) for (Ai,Wi) in zip(A,W)]
end

# force matrices to be Hermitian (ishermitian) by forcing numerical symmetry and adding a regularization term
function sampleAW(agent; n_samples=1)
    r = 1e-6
    Ω_inv = (inv(agent.Ω) + inv(agent.Ω)') / 2 + r*I
    Λ_inv = (inv(agent.Λ) + inv(agent.Λ)') / 2 + r*I
    A = zeros(agent.Dx, agent.Dy, n_samples)
    W = rand(Wishart(agent.ν, Ω_inv), n_samples)
    for (i, Wi) in enumerate(W)
        Wi_inv = (inv(Wi) + inv(Wi)') / 2 + r*I
        A[:,:,i] = rand(MatrixNormal(agent.M, Λ_inv, Wi_inv))
    end
    A = [Matrix(slice) for slice in eachslice(A, dims=3)]
    # symmetrize and regularize W again...
    #W = (W + W') / 2 + r*I
    return [(Ai,Wi) for (Ai,Wi) in zip(A,W)]
end

function EFE(agent::MARXAgent, controls::AbstractVector{T}) where T
#function EFE(agent::MARXAgent, controls)
    "Expected Free Energy"
    N = 10
    r = 1e-6

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer

    println("[ ] Sampling")
    # sampling of a joint over the future predictions for time horizon of 2
    AW = sampleAW(agent, n_samples=N)
    # TODO thorizon times D_y
    # TODO generalize to thorizon > 2
    μ = zeros(2*agent.Dy)
    Σ = zeros(2*agent.Dy, 2*agent.Dy)
    for i in 1:N
        A, W = AW[i]
        mvn_mean = zeros(2*agent.Dy)
        mvn_cov = zeros(2*agent.Dy, 2*agent.Dy)

        # Assuming y buffer is being put at the end of the buffer, then the block of A corresponding to the future output are thorizon-1 last rows
        A2 = A[end-1:end,:]
        W_inv = (inv(W) + inv(W)') / 2 + r*I
        Σi = [W_inv W_inv*A2'; A2*W_inv A2*W_inv*A2' + W_inv]
        Σi = (Σi + Σi') / 2 + r*I

        # x_t = x_{k+1} contains u_{k+1} and y_k
        #u_t = controls[(t-1)*agent.Du+1:t*agent.Du]
        #u_t = controls[1:2]
        ubuffer = backshift(ubuffer, controls[1:2])
        x_t = [ubuffer[:]; ybuffer[:]]
        py_t = A'x_t
        println("type(x_t)")
        for (xi, x) in enumerate(x_t)
            println(xi, " ", x)
        end

        yp1buffer = backshift(ybuffer, py_t)
        up1buffer = backshift(ubuffer, controls[3:4])
        x_tp1 = [ubuffer[:]; ybuffer[:]]

        μi = [py_t; A'x_tp1]

        # moving average
        μ += (μi - μ)./i
        Σ += (Σi - Σ)./i
    end
    println("[X] Sampling")

    # TODO: forgot to sum
    # expectation of outputs via sampling of N samples
    #μ = μs[:,i] ./ N
    #Σ = Σs[:,i] ./ N
    pys = MvNormal(μ, Σ)
    println(pys)

    # TODO use pys as joint posterior predictive for EFE
    J = rand(1)[1]
    if false
        for t in 1:agent.thorizon
    
            # Current control
            u_t = controls[(t-1)*agent.Du+1:t*agent.Du]
            
            # Update control buffer
            ubuffer = backshift(ubuffer, u_t)
            x_t = [ubuffer[:]; ybuffer[:]]
    
            # Calculate and accumulate EFE
            J += mutualinfo(agent, x_t) + crossentropy(agent, x_t) # + u_t'*agent.Υ*u_t
            
            # Update previous 
            _, m_y, _ = posterior_predictive(agent, x_t)
            ybuffer = backshift(ybuffer, m_y)        
        end
    J += mutualinfo(agent, x_t) + crossentropy(agent, x_t)
    end
    return J
end

function minimizeEFE(agent::MARXAgent; u_0=nothing, time_limit=10, verbose=false, control_lims::Tuple=(-Inf,Inf))
    "Minimize EFE objective and return policy."

    if isnothing(u_0); u_0 = 1e-8*randn(agent.thorizon); end
    opts = Optim.Options(time_limit=time_limit, 
                         show_trace=verbose, 
                         allow_f_increases=true, 
                         g_tol=1e-12, 
                         show_every=10,
                         iterations=10_000)

    # Objective function
    J(u) = EFE(agent, u)

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

function multigamma(p,a)
    result = π^(p*(p-1)/4)
    for j = 1:p 
        result *= gamma(a + (1-j)/2)
    end
    return result
end

function logmultigamma(p,a)
    result = p*(p-1)/4*log(π)
    for j = 1:p 
        result += loggamma(a + (1-j)/2)
    end
    return result
end

end
