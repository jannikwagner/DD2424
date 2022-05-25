using LinearAlgebra, Plots, DifferentialEquations

D = [.5 0; 0 -.9]/5
S = [5 1; 1 5]
A = S*D*S^(-1)

xs = -1.25:0.25:1.25
ys = -1.25:0.25:1.25

df(x, y) = A*[x,y]

function ODEfunc!(du,u,p,t)
    du[1] = A[1,1]*u[1]+A[1,2]*u[2];
    du[2] = A[2,1]*u[1]+A[2,2]*u[2];
end


tspan = (0.0,25.0)
tspanLong = (-0.1,100.0)
u01 = [-0.1;1.1]
u02 = [0.3;1.1]
u03 = [-0.12;-1.1]
u04 = [-0.315;-1.1]
u05 = [0.18;1.1]
u06 = [0.48;1.1]
u07 = [0.1;-1.1]
u08 = [-0.55;-1.1]
u09 = [-0.4;1.1]
u010 = [0.7;1.1]
u011 = [0.4;-1.1]
u012 = [-0.75;-1.1]
prob1 = ODEProblem(ODEfunc!,u01,tspanLong)
prob2 = ODEProblem(ODEfunc!,u02,tspanLong)
prob3 = ODEProblem(ODEfunc!,u03,tspanLong)
prob4 = ODEProblem(ODEfunc!,u04,tspanLong)
prob5 = ODEProblem(ODEfunc!,u05,tspanLong)
prob6 = ODEProblem(ODEfunc!,u06,tspanLong)
prob7 = ODEProblem(ODEfunc!,u07,tspanLong)
prob8 = ODEProblem(ODEfunc!,u08,tspanLong)
prob9 = ODEProblem(ODEfunc!,u09,tspanLong)
prob10 = ODEProblem(ODEfunc!,u010,tspanLong)
prob11 = ODEProblem(ODEfunc!,u011,tspanLong)
prob12 = ODEProblem(ODEfunc!,u012,tspanLong)
#=
sol1 = solve(prob1)
sol2 = solve(prob2)
sol3 = solve(prob3)
sol4 = solve(prob4)
sol5 = solve(prob5)
sol6 = solve(prob6)
sol7 = solve(prob7)
sol8 = solve(prob8)
sol9 = solve(prob9)
sol10 = solve(prob10)
sol11 = solve(prob11)
sol12 = solve(prob12)
discrete_t = 
plot(sol1,vars=(1,2),xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
plot!(sol2,vars=(1,2),xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2)
plot!(sol3,vars=(1,2),xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2)
plot!(sol4,vars=(1,2),xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2)
plot!(sol5,vars=(1,2),xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2)
plot!(sol6,vars=(1,2),xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2)
plot!(sol7,vars=(1,2),xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2)
plot!(sol8,vars=(1,2),xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2)
plot!(sol9,vars=(1,2),xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2)
plot!(sol10,vars=(1,2),xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2)
plot!(sol11,vars=(1,2),xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2)
plot!(sol12,vars=(1,2),xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2)
=#

t_red = 1:5:21
sol1x = Array(solve(prob1))[1,[1,3,5,7,10]]
sol1y = Array(solve(prob1))[2,[1,3,5,7,10]]
sol2x = Array(solve(prob2))[1,1:3:15]
sol2y = Array(solve(prob2))[2,1:3:15]
sol3x = Array(solve(prob3))[1,1:3:15]
sol3y = Array(solve(prob3))[2,1:3:15]
sol4x = Array(solve(prob4))[1,[1,4,6,10,12]]
sol4y = Array(solve(prob4))[2,[1,4,6,11,12]]
sol5x = Array(solve(prob5))[1,[1,4,7,11,19]]
sol5y = Array(solve(prob5))[2,[1,4,7,11,19]]
sol6x = Array(solve(prob6))[1,[1,4,6,8,11]]
sol6y = Array(solve(prob6))[2,[1,4,6,8,11]]
sol7x = Array(solve(prob7))[1,[1,4,6,8,11]]
sol7y = Array(solve(prob7))[2,[1,4,6,8,11]]
sol8x = Array(solve(prob8))[1,[1,4,6,8,11]]
sol8y = Array(solve(prob8))[2,[1,4,6,8,11]]
sol9x = Array(solve(prob9))[1,[1,4,6]]
sol9y = Array(solve(prob9))[2,[1,4,6]]
sol10x = Array(solve(prob10))[1,[1,4,6]]
sol10y = Array(solve(prob10))[2,[1,4,6]]
sol11x = Array(solve(prob11))[1,[1,4,5,7]]
sol11y = Array(solve(prob11))[2,[1,4,5,7]]
sol12x = Array(solve(prob12))[1,[1,4,6,7]]
sol12y = Array(solve(prob12))[2,[1,4,6,7]]

sol4y[3] = sol4y[3] + 0.0
sol4y[4] = sol4y[4] - 0.03
sol4x[3] = sol4x[3] + 0.05
sol4x[4] = sol4x[4] + 0.2

plot(sol1x,sol1y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
scatter!(sol1x,sol1y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
plot!(sol2x,sol2y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
scatter!(sol2x,sol2y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
plot!(sol3x,sol3y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
scatter!(sol3x,sol3y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
plot!(sol4x,sol4y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
scatter!(sol4x,sol4y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
plot!(sol5x,sol5y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
scatter!(sol5x,sol5y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
plot!(sol6x,sol6y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
scatter!(sol6x,sol6y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
plot!(sol7x,sol7y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
scatter!(sol7x,sol7y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
plot!(sol8x,sol8y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
scatter!(sol8x,sol8y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
plot!(sol9x,sol9y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
scatter!(sol9x,sol9y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
plot!(sol10x,sol10y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
scatter!(sol10x,sol10y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
plot!(sol11x,sol11y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
scatter!(sol11x,sol11y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
plot!(sol12x,sol12y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
scatter!(sol12x,sol12y,xlim=(-1.1,1.1),ylim=(-1.1,1.1),label="", color=:blue, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))


xxs = [x for x in xs for y in ys]
yys = [y for x in xs for y in ys]
quiver!(xxs, yys, quiver=df, color=:gray, axis=([], false),aspect_ratio=2, dpi=500, size=(200,400))
#savefig( "discreteflow.png")
#savefig( "continuousflow.png")