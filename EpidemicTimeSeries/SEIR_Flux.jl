using DifferentialEquations, Plots, Flux, DiffEqFlux

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
γ = 1/18
R₀ = 3.0
σ = 1/5.2 # parameters
p_0 = [γ,R₀,σ]

tspan = (0.0, 350.0)  # ≈ 350 days
prob = ODEProblem(EpMod, x_0, tspan, p_0)

# Solve the problem with an ODE solver
sol = solve(prob, Tsit5())
sol2 = solve(prob,Tsit5(),saveat=0.1)
# plot the solution
plot(sol, labels = ["s" "e" "i" "r"], title = "SEIR Dynamics", lw = 2, xlabel = "t")
t = 0:0.1:350
tshort = 1:100:3500
A = sol2[3,tshort] # length 101 vector
A = A + randn(length(tshort))/100
scatter!(0:10:350,A)


# Initial guess
p = [0.1,5,0.1]
params = Flux.params(p)

# Prediction function
function predict_rd() # Our 1-layer "neural network"
  solve(prob,Tsit5(),p=p,saveat=0.1)[3,:] # override with new parameters
end

# loss function: MSE
loss_rd() = sum(abs2,predict_rd()[tshort] .-  A) 

# Perform training
data = Iterators.repeated((), 200)
opt = ADAM()
cb = function () #callback function to observe training
  display(loss_rd())
  # using `remake` to re-create our `prob` with current parameters `p`
  #display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,10)))
  cur_pred = predict_rd()
  pl = scatter(0:10:349,A,label="Data")
  plot!(pl,solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,1.1), labels = ["S" "E" "I" "R"], legend=:bottomleft,dpi=500)
  #scatter!(pl,t,cur_pred[1,:],label="prediction")
  display(plot(pl))
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_rd, params, data, opt, cb = cb)