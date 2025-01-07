using LinearAlgebra
using DSP


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

function angle2pos(z; l=1.0)
    "Map angle to Cartesian position"
    return (l*sin(z), -l*cos(z))
end

function lowpass(x::Vector; order=1, Wn=1.0, fs=1.0)
    "Extract AR coefficients based on low-pass Butterworth filter"

    dfilter = digitalfilter(Lowpass(Wn; fs=fs), Butterworth(order))
    
    tf = convert(PolynomialRatio, dfilter)
    b = coefb(tf)
    a = coefa(tf)

    return filt(b,a, x), a,b
end

function polar2cart(θ::Float64, r::Float64)
    x =  sin(θ)
    y = -cos(θ)
    return r*[x,y]
end

function cart2polar(x::Float64, y::Float64)
    r = sqrt(x^2 + y^2)
    θ = atan(-y, x)  # Use atan2 to get the angle in the correct quadrant
    return (θ, r)
end