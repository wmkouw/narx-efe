module NARXsystem

using LinearAlgebra

export NARXsys, update!, orders

mutable struct NARXsys

    order_inputs   ::Integer
    order_outputs  ::Integer
    input_buffer   ::Vector{Float64}
    output_buffer  ::Vector{Float64}
    basis_function ::Function
    coefficients   ::Vector{Float64}
    observation    ::Float64
    mnoise_sd      ::Float64 # Measurement noise standard deviation
    input_lims     ::Tuple{Float64,Float64}

    function NARXsys(coefficients::Vector{Float64}, 
                     basis_function::Function, 
                     mnoise_sd::Float64; 
                     order_inputs::Integer = 1, 
                     order_outputs::Integer = 1,
                     input_lims::Tuple{Float64,Float64} = (-Inf, Inf))

        input_buffer  = zeros(1+order_inputs)
        output_buffer = zeros(order_outputs)
        init_observation = 0.0

        return new(order_inputs, 
                   order_outputs,
                   input_buffer,
                   output_buffer,
                   basis_function,
                   coefficients,
                   init_observation,
                   mnoise_sd,
                   input_lims)
    end
end

function update!(sys::NARXsys, input::Float64)

    # Update buffer with previous observation
    sys.output_buffer = backshift(sys.output_buffer, sys.observation)

    # Update input buffer
    clamped_input = clamp.(input, sys.input_lims...)
    sys.input_buffer = backshift(sys.input_buffer, clamped_input)

    # Generate new observation
    ϕ = sys.basis_function([sys.output_buffer; sys.input_buffer])
    sys.observation  = dot(sys.coefficients, ϕ) + sys.mnoise_sd*randn()    
    
end

function orders(sys::NARXsys)
    return (sys.order_inputs, sys.order_outputs)
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

end;
