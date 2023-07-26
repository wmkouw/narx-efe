module Pendulums

using LinearAlgebra

export SPendulum, params, dzdt, RK4, update!, emit!, step!


mutable struct SPendulum
    "Single Pendulum"

    state       ::Vector{Float64}
    sensor      ::Float64

    Δt          ::Float64
    
    mass        ::Float64
    length      ::Float64
    damping     ::Float64

    mnoise_sd   ::Float64 # measurement noise standard deviation

    function SPendulum(init_state, mass, length, damping, mnoise_sd; Δt::Float64=1.0)
        
        init_sensor = init_state[1] + mnoise_sd*randn()
        return new(init_state, init_sensor, Δt, mass, length, damping, mnoise_sd)
    end
end

function params(sys::SPendulum)
    return (sys.mass, sys.length, sys.damping)
end

function dzdt(state::Vector, u::Float64, params::Tuple)
    
    mass, length, damping = params
    return [state[2]; -9.81/length*sin(state[1]) - damping*length*state[2] + 1/mass*u]    
end

function RK4(sys::SPendulum, u::Float64)
    
    K1 = dzdt(sys.state              , u, params(sys))
    K2 = dzdt(sys.state + K1*sys.Δt/2, u, params(sys))
    K3 = dzdt(sys.state + K2*sys.Δt/2, u, params(sys))
    K4 = dzdt(sys.state + K3*sys.Δt  , u, params(sys))
    
    return sys.Δt/6 * (K1 + 2K2 + 2K3 + K4)
end

function update!(sys::SPendulum, u::Float64)
    sys.state  = sys.state + RK4(sys, u)
end

function emit!(sys::SPendulum)
    sys.sensor = sys.state[1] + sys.mnoise_sd * randn()
end

function step!(sys::SPendulum, u::Float64)
    update!(sys, u)
    emit!(sys)
end

end