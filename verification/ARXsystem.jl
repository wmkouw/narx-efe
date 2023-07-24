module ARXsystem

using LinearAlgebra

export ARXsys, update!, orders

mutable struct ARXsys

    order_inputs   ::Integer
    order_outputs  ::Integer
    input_buffer   ::Vector{Float64}
    output_buffer  ::Vector{Float64}

    coefficients   ::Vector{Float64}
    
    observation    ::Float64

    # Measurement noise standard deviation
    mnoise_sd      ::Float64 

    function ARXsys(coefficients, mnoise_sd; order_inputs=1, order_outputs=1)

        input_buffer  = zeros(order_inputs)
        output_buffer = zeros(order_outputs)
        init_observation = 0.0

        return new(order_inputs, 
                   order_outputs,
                   input_buffer,
                   output_buffer,
                   coefficients,
                   init_observation,
                   mnoise_sd)
    end
end

function update!(sys::ARXsys, input::Float64)

    # Update buffer with previous observation
    sys.output_buffer = backshift(sys.output_buffer, sys.observation)

    # Update input buffer
    sys.input_buffer = backshift(sys.input_buffer, input)

    # Generate new observation
    sys.observation  = dot(sys.coefficients, [1.0; sys.output_buffer; sys.input_buffer]) + sys.mnoise_sd*randn()    
    
end

function orders(sys::ARXsys)
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

function backshift(M::AbstractMatrix, a::Number)
    return diagm(backshift(diag(M), a))
end

end;
