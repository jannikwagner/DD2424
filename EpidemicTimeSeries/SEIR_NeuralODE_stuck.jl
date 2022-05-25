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
tshort = 1:100:3500
ode_data = sol[3,tshort]
plt = scatter(t[tshort],ode_data, label = "data")

datasize = length(ode_data) # Number of data points
tspan = (0.0f0, 350.0f0) # Time range
tsteps = range(tspan[1], tspan[2], length = datasize) # Split time range into equal steps for each data point



# Make a neural net with a NeuralODE layer
dudt2 = FastChain(FastDense(1, 50, swish), # Multilayer perceptron for the part we don't know
                  FastDense(50, 20, swish),
                  FastDense(20, 1))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

# Array of predictions from NeuralODE with parameters p starting at initial condition u0
u01 = Float32[0.0]
function predict_neuralode(p)
  Array(prob_neuralode(u01, p))
end

function loss_neuralode(p)
    pred = Array(predict_neuralode(p))
    loss = sum(abs2, ode_data .- pred) # Just sum of squared error
    return loss, pred
end

# Callback function to observe training
callback = function (p, l, pred; doplot = false)
  display(l)
  # plot current prediction against data
  predics = Array(pred)
  print("\n")
  #plt = scatter(tsteps, ode_data, label = "data")
  #scatter!(plt, tsteps,   predics = Array(pred), label = "prediction")
  if doplot
    #display(plot(plt))
    #print(predict_neuralode(p)) 
    #open("intermediate_results_2wave.txt", "a") do io
    #  writedlm(io, pred[1,:])
    #end  
  end
  return false
end

# Parameters are prob_neuralode.p
result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,cb = callback,maxiters=100)
l, û = loss_neuralode(result_neuralode.minimizer)
plt = scatter(tsteps, ode_data, label = "data")
scatter!(plt, tsteps, Array(û)[1:end], label = "prediction")