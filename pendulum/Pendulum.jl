using LinearAlgebra

export Pendulum, params, dzdt, RK4, update!, emit!, step!

mutable struct Pendulum

    state       ::Vector{Float64}
    sensor      ::Float64
    
    mass        ::Float64
    length      ::Float64
    damping     ::Float64

    mnoise_sd   ::Float64 # measurement noise standard deviation

    function Pendulum(init_state, mass, length, damping, mnoise_sd)
        
        init_sensor = init_state[1] + mnoise_sd*randn()
        return new(init_state, init_sensor, mass, length, damping, mnoise_sd)
    end
end

function params(sys::Pendulum)
    return (sys.mass, sys.length, sys.damping)
end

function dzdt(state::Vector, u::Float64, params::Tuple)
    
    mass, length, damping = params
    return [state[2]; -9.81/length*sin(state[1]) - damping*length*state[2] + 1/mass*u]    
end

function RK4(sys::Pendulum, u::Float64; Δt::Float64=1.0)
    
    K1 = dzdt(sys.state          , u, params(sys))
    K2 = dzdt(sys.state + K1*Δt/2, u, params(sys))
    K3 = dzdt(sys.state + K2*Δt/2, u, params(sys))
    K4 = dzdt(sys.state + K3*Δt  , u, params(sys))
    
    return Δt/6 * (K1 + 2K2 + 2K3 + K4)
end

function update!(sys::Pendulum, u::Float64; Δt::Float64 = 1.0)
    sys.state  = sys.state + RK4(sys, u, Δt=Δt)
end

function emit!(sys::Pendulum)
    sys.sensor = sys.state[1] + sys.mnoise_sd * randn()
end

function step!(sys::Pendulum, u::Float64; Δt::Float64 = 1.0)
    update!(sys, u, Δt=Δt)
    emit!(sys)
end