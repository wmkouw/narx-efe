module Pendulums

using LinearAlgebra

export SPendulum, DPendulum, params, dzdt, RK4, update!, emit!, step!

abstract type Pendulum end

mutable struct SPendulum <: Pendulum
    "Single Pendulum"

    state       ::Vector{Float64}
    sensor      ::Float64

    Δt          ::Float64
    
    mass        ::Float64
    length      ::Float64
    damping     ::Float64

    mnoise_sd   ::Float64 # measurement noise standard deviation

    function SPendulum(;init_state::Vector{Float64}=zeros(2), 
                        mass::Float64=1.0, 
                        length::Float64=1.0, 
                        damping::Float64=0.0, 
                        mnoise_sd::Float64=1.0,
                        Δt::Float64=1.0)
        
        init_sensor = init_state[1] + mnoise_sd*randn()
        return new(init_state, init_sensor, Δt, mass, length, damping, mnoise_sd)
    end
end

mutable struct DPendulum <: Pendulum
    "Double Pendulum"

    state       ::Vector{Float64}
    sensor      ::Vector{Float64}

    Δt          ::Float64
    
    mass        ::Vector{Float64}
    length      ::Vector{Float64}
    damping     ::Float64

    mnoise_S    ::Matrix{Float64}

    function DPendulum(;init_state::Vector{Float64}=zeros(4), 
                        mass::Vector{Float64}=[1.,1.], 
                        length::Vector{Float64}=[1.,1.], 
                        damping::Float64=0.0, 
                        mnoise_S::Matrix{Float64}=diagm(ones(2)),
                        Δt::Float64=1.0)
        
        init_sensor = init_state[1:2] + cholesky(mnoise_S).L*randn(2)
        return new(init_state, init_sensor, Δt, mass, length, damping, mnoise_S)
    end
end

params(sys::Pendulum) = (sys.mass, sys.length, sys.damping)

function dzdt(sys::SPendulum, Δstate::Vector, u::Float64)

    z = sys.state + Δstate
    
    mass, length, damping = params(sys)
    return [z[2]; -9.81/length*sin(z[1]) - damping*length*z[2] + 1/mass*u] 
end

function RK4(sys::Pendulum, u::Float64)
    
    K1 = dzdt(sys, zeros(2)   , u)
    K2 = dzdt(sys, K1*sys.Δt/2, u)
    K3 = dzdt(sys, K2*sys.Δt/2, u)
    K4 = dzdt(sys, K3*sys.Δt  , u)
    
    return sys.Δt/6 * (K1 + 2K2 + 2K3 + K4)
end

function update!(sys::Pendulum, u)
    sys.state  = sys.state + RK4(sys, u)
end

function emit!(sys::SPendulum)
    sys.sensor = sys.state[1] + sys.mnoise_sd * randn()
end
function emit!(sys::DPendulum)
    sys.sensor = sys.state[1:2] + cholesky(sys.mnoise_S).L * randn()
end

function step!(sys::Pendulum, u)
    update!(sys, u)
    emit!(sys)
end         

end