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

function trackbot(tk)

    xl = extrema(y_sim[1,:])
    yl = extrema(y_sim[2,:])
    ttime = round(tk*Δt, digits=1)
    plot(title="time = $ttime sec", xlims=xl, ylims=yl)

    scatter!([z_0[1]], [z_0[2]], label="start", color="green", markersize=5)
    scatter!([mean(goal)[1]], [mean(goal)[2]], label="goal", color="red", markersize=5)
    covellipse!(mean(goal), cov(goal), n_std=1., linewidth=3, fillalpha=0.01, linecolor="red", color="red")
    scatter!([y_sim[1,tk]], [y_sim[2,tk]], alpha=0.3, label="observations", color="black")
    plot!([z_sim[1,tk]], [z_sim[2,tk]], marker=:star5, markersize=5, label="system path", color="blue")
   
    for kk = 1:len_horizon
        covellipse!(y_pln[1][tk,:,kk], y_pln[2][tk,:,:,kk]/100, linewidth=0, n_std=1, fillalpha=0.1, color="orange")
    end
    plot!(y_pln[1][tk,1,:], y_pln[1][tk,2,:], color="orange", label="planning")

end