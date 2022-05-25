using DifferentialEquations, Plots, DelimitedFiles, Flux

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

#=
N = 10^7 # Population
i_0 = 2000/N                # Initially infected
e_0 = 4.0 * i_0             # Initially exposed
s_0 = 1.0 - i_0 - e_0       # Initially susceptible
r_0 = 0.0                   # Initially recovered
x_0 = [s_0, e_0, i_0, r_0]  # initial state of the system
γ = 1/18
R₀ = 3.0
σ = 1/5.2 
p_0 = [γ,R₀,σ] # parameters
=#
N = 10^7
i_0 = 1E-7                  # 33 = 1E-7 * 330 million population = initially infected
e_0 = 4.0 * i_0             # 132 = 1E-7 *330 million = initially exposed
s_0 = 1.0 - i_0 - e_0
r_0 = 0.0
x_0 = [s_0, e_0, i_0, r_0]  # initial condition
γ = 1/5
R₀ = 3.0
σ = 1/5.2 # parameters
p_0 = [γ,R₀,σ]


dat = readdlm("covid.txt",',');
time_i = dat[:,1]; # times
Infected = 50*dat[:,2]/N; # Normalize data
Infected[2:end] = Infected[2:end] 
#scatter(time_i,Infected,label="Infected")


# Solve the problem with an ODE solver
tspan = (0.0, maximum(time_i))  # ≈ 170 days
prob = ODEProblem(EpMod, x_0, tspan, p_0)
sol = solve(prob, Tsit5())
sol2 = solve(prob,Tsit5(),saveat=0.1)
# plot the solution
plot(sol, labels = ["s" "e" "i" "r"], title = "SEIR Dynamics", lw = 2, xlabel = "t")
tshort_inds = (Int.(time_i)*10).+1
tshort = Int.(time_i).+1
A = sol2[3,tshort_inds] # length 101 vector
A = A + randn(length(tshort))/1000
A = Infected;
scatter!(tshort,A)



# Initial guess
p = [0.17,5.7,0.170]
params = Flux.params(p)

# Prediction function
function predict_rd() # Our 1-layer "neural network"
  solve(prob,Tsit5(),p=p,saveat=0.1,maxiters=1e7)[3,:] # override with new parameters
end

# loss function: MSE
loss_rd() = sum(abs2,predict_rd()[tshort_inds] .-  A) 

# Perform training
data = Iterators.repeated((), 200)
opt = ADAM()
cb = function () #callback function to observe training
  display(loss_rd())
  # using `remake` to re-create our `prob` with current parameters `p`
  #display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,10)))
  cur_pred = predict_rd()
  pl = scatter(tshort,A,label="Data",yticks = ([0:0.05:0.25;], ["0", "10000", "20000", "30000", "40000", "50000", "60000"]))
  plot!(pl,solve(remake(prob,p=p),Tsit5(),saveat=0.1,maxiters=1e7),vars=(3),ylim=(0,0.25), labels = "ODE approximation (SEIR)", legend=:topright, dpi=500, color="blue")
  #scatter!(pl,t,cur_pred[1,:],label="prediction")
  display(plot(pl))
end

# Display the ODE with the initial parameter values.
cb()
#=
numEpochs = 300
k = 10
train_loader = Flux.Data.DataLoader((data, t), batchsize = k)
=#
Flux.train!(loss_rd, params, data, opt, cb = cb)