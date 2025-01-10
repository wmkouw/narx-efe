using LinearAlgebra
using Distributions
using SpecialFunctions
using Plots
default(label="", linewidth=2)

### Sampling

# Component distributions
mx,vx = (0.0,1.0)
my,vy = (0.0,1.0)
px = Normal(mx,sqrt(vx))
py = Normal(my,sqrt(vy))

N = 100_000

x = rand(px,N)
y = rand(py,N)
z = x .* y

mdz_hat = median(z)
mz_hat  = mean(z)
vz_hat  = var(z)

# Moment matching
qmz = mx*my
qvz = (vx + mx^2)*(vy + my^2) - mx^2*my^2

diff_mz = abs(mz_hat - qmz)
diff_vz = abs(vz_hat - qvz)
println("| empirical mean - moment-matched mean | = $diff_mz")
println("| empirical variance - moment-matched variance | = $diff_vz")

### Visual comparison

zr = range(-5,stop=5, length=100)

# Histogram
histogram(z, bins=zr, normalize=:pdf, xlims=extrema(zr))

# Bessel function
pz(z) = besselk.(0, abs(z))./π
plot!(zr, pz.(zr), lw=3, color="red", xlims=extrema(zr))

# Gaussian with moment matching (EP)
qz(z) = pdf(Normal(qmz,sqrt(qvz)),z)
plot!(zr, qz.(zr), lw=3, color="purple", xlims=extrema(zr))

