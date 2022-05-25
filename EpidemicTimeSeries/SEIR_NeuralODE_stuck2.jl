using DiffEqFlux, DifferentialEquations, Plots, GalacticOptim, DelimitedFiles

# Define ODE
function EpMod(x, p, t)
  s, e, i, r = x
  γ, R₀, σ = p
  return [-γ*R₀*s*i;       # ds/dt = -γR₀si
           γ*R₀*s*i -  σ*e;# de/dt =  γR₀si -σe
           σ*e - γ*i;      # di/dt =         σe -γi
                 γ*i;      # dr/dt =             γi
          ]
end

i_0 = 1E-7                  # 33 = 1E-7 * 330 million population = initially infected
e_0 = 4.0 * i_0             # 132 = 1E-7 *330 million = initially exposed
s_0 = 1.0 - i_0 - e_0
r_0 = 0.0
x_0 = [s_0, e_0, i_0, r_0]  # initial condition
γ = 1/10
R₀ = 3.0
σ = 1/5.2 # parameters
p_0 = [γ,R₀,σ]

tspan = (0.0, 350.0)  # ≈ 350 days
prob_E = ODEProblem(EpMod, x_0, tspan, p_0)

# Solve the problem with an ODE solver
sol = solve(prob_E,Tsit5(),saveat=0.1)
t = 0:0.1:350
tshort = 1:100:500
ode_data = sol[3,tshort]
u = ode_data
plt = scatter(t[tshort],ode_data, label = "data")

datasize = length(ode_data) # Number of data points
tsteps = range(tspan[1], tspan[2], length = datasize) # Split time range into equal steps for each data point
tspan = (minimum(t), maximum(t))
dudt2 = FastChain(FastDense(1, 32, swish),
                FastDense(32, 64, swish),
                FastDense(64, 32, swish),
                FastDense(32, 1))

node = NeuralODE(dudt2, tspan, Vern7(), saveat = t[tshort], 
                  abstol = 1e-6, reltol = 1e-6)

u0 = u[1:1]
predict(p) = node(u0, p)
loss(p) = begin
  û = reshape(predict(p),:)
  #l = Flux.mse(û, u)
  l = sum(abs2,u-û) 
  return l, û
end

result = DiffEqFlux.sciml_train(loss, node.p, ADAM(),maxiters = 500)
l, û = loss(result.minimizer)
plt = scatter(t[tshort], u, label = "Data")
scatter!(plt, t[tshort], reshape(û, :), label = "Prediction")

