module Pendulums

using LinearAlgebra

export SPendulum, DPendulum, params, dzdt, RK4, update!, emit!, step!, sim_trajectory

abstract type Pendulum end

mutable struct SPendulum <: Pendulum
    "Single Pendulum"

    state       ::Vector{Float64}
    sensor      ::Float64
    torque      ::Float64
    torque_lims ::Tuple{Float64,Float64}
    Δt          ::Float64
    mass        ::Float64
    length      ::Float64
    damping     ::Float64
    mnoise_sd   ::Float64 # measurement noise standard deviation

    function SPendulum(;init_state::Vector{Float64}=zeros(2), 
                        torque_lims::Tuple{Float64,Float64}=(-1.0, 1.0),
                        mass::Float64=1.0, 
                        length::Float64=1.0, 
                        damping::Float64=0.0, 
                        mnoise_sd::Float64=1.0,
                        Δt::Float64=1.0)
        
        init_sensor = init_state[1] + mnoise_sd*randn()
        return new(init_state, init_sensor, 0.0, torque_lims, Δt, mass, length, damping, mnoise_sd)
    end
end

mutable struct DPendulum <: Pendulum
    "Double Pendulum"

    state       ::Vector{Float64}
    sensor      ::Vector{Float64}
    torque      ::Vector{Float64}
    torque_lims ::Tuple{Float64,Float64}
    Δt          ::Float64
    mass        ::Vector{Float64}
    length      ::Vector{Float64}
    damping     ::Float64
    mnoise_S    ::Matrix{Float64}

    function DPendulum(;init_state::Vector{Float64}=zeros(4), 
                        torque_lims::Tuple{Float64,Float64}=(-1.0, 1.0),
                        mass::Vector{Float64}=[1.,1.], 
                        length::Vector{Float64}=[1.,1.], 
                        damping::Float64=0.0, 
                        mnoise_S::Matrix{Float64}=diagm(ones(2)),
                        Δt::Float64=1.0)
        
        init_sensor = init_state[1:2] + cholesky(mnoise_S).L*randn(2)
        return new(init_state, init_sensor, zeros(2), torque_lims, Δt, mass, length, damping, mnoise_S)
    end
end

params(sys::Pendulum) = (sys.mass, sys.length, sys.damping)

function dzdt(sys::SPendulum, u::Float64; Δstate::Vector=zeros(2))
    "Equations of motion of single pendulum"
    z = sys.state + Δstate 
    mass, length, damping = params(sys)
    return [z[2]; -9.81/length*sin(z[1]) - damping*length*z[2] + 1/mass*u] 
end

function dzdt(sys::DPendulum, u::Vector{Float64}; Δstate::Vector=zeros(4), κ::Float64=0.0, gravity::Float64=9.81)
    "Equations of motion of double pendulum"
    
    z = sys.state + Δstate 

    (m1,m2), (l1,l2), damping = params(sys)

    # Shorthand notation
    Ja = 1/3*m1*l1^2 + m2*l1^2
    Jb = 1/3*m2*l2^2
    Jx = 1/2*m2*l1*l2
    μ1 = (m1/2 + m2)*gravity*l1
    μ2 = 1/2*m2*gravity*l2
    
    # Inverse mass (inertia matrix)
    Mi = 1/(Ja*Jb - Jx*cos(z[1] - z[2])*Jx*cos(z[1] - z[2]))*[Jb -Jx*cos(z[1] - z[2]);-Jx*cos(z[1] - z[2]) Ja]
    
    # Equations of motion
    ddθ1 = -Jx*sin(z[1] - z[2])*z[4]^2 - μ1*sin(z[1]) + κ*sin(z[2] - z[1]) + u[1]
    ddθ2 =  Jx*sin(z[1] - z[2])*z[3]^2 - μ2*sin(z[2]) + κ*sin(z[2] - z[1]) + u[2]
    ddθ = Mi*[ddθ1, ddθ2]
    
    return [z[3]; z[4]; ddθ[1]; ddθ[2]]
end

function RK4(sys::Pendulum, u)
    
    K1 = dzdt(sys, u)
    K2 = dzdt(sys, u, Δstate=K1*sys.Δt/2)
    K3 = dzdt(sys, u, Δstate=K2*sys.Δt/2)
    K4 = dzdt(sys, u, Δstate=K3*sys.Δt  )
    
    return sys.Δt/6 * (K1 + 2K2 + 2K3 + K4)
end

function update!(sys::Pendulum, u)
    sys.torque = clamp.(u, sys.torque_lims...)
    sys.state = sys.state + RK4(sys, sys.torque)
end

function emit!(sys::SPendulum)
    sys.sensor = sys.state[1] + sys.mnoise_sd * randn()
end
function emit!(sys::DPendulum)
    sys.sensor = sys.state[1:2] + cholesky(sys.mnoise_S).L * randn(2)
end

function step!(sys::Pendulum, u)
    update!(sys, u)
    emit!(sys)
end         

function sim_trajectory(sys::Pendulum, policy)
    "Simulate trajectory of pendulum for a given policy"

    time_horizon = length(policy)
    state_dim = length(sys.state)
    
    trajectory = zeros(state_dim, time_horizon)
    state_tmin1 = sys.state
    for t in 1:time_horizon
        trajectory[:,t] = state_tmin1 + sys.Δt*dzdt(sys, policy[t], Δstate=state_tmin1-sys.state)
        state_tmin1 = trajectory[:,t]
    end
    return trajectory
end

end