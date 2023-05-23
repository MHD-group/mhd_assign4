#=
	Created On  : 2023-04-03 00:24
    Copyright © 2023 Junyi Xu <jyxu@mail.ustc.edu.cn>

    Distributed under terms of the MIT license.
=#

using PyCall
using LaTeXStrings

# %%
@pyimport matplotlib.pyplot as plt
@pyimport matplotlib # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
matplotlib.rc("font", size=9)

const γ = 1.4
const β = 2.0

# %%

function w2U(w::Vector)::Vector
	ρ = w[1]
	p = w[2]
	vx = w[3]
	vy = w[4]
	vz = w[5]
	Hy = w[6]
	Hz = w[7]
	return [ρ
			ρ*(vx^2 + vy^2 +vz^2) + Hy^2 + Hz^2 + β*p/(γ-1)
			ρ*vx
			ρ*vy
			ρ*vz
			Hy
			Hz ]
end

function w2U(w::Matrix)::Matrix
	U = similar(w)
	for l = 1:size(U, 2)
		U[:, l] .= w[:, l] |> w2U
	end
	return U
end

function w2A(w::Vector)::Matrix
	ρ = w[1]
	E = w[2]
	mx = w[3]
	my = w[4]
	mz = w[5]
	Hy = w[6]
	Hz = w[7]
	A = [0    0    1    0    0    0    0
	(2*Hx*ρ*(Hy*my+Hz*mz)+mx*(ρ*(-γ*E+(γ-2)*Hy^2+(γ-2)*Hz^2)+2*(γ-1)*my^2+2*(γ-1)*mz^2)+2*(γ-1)*mx^3)/ρ^3    (γ*mx)/ρ    -((-γ*E*ρ+(γ-2)*Hy^2*ρ+(γ-2)*Hz^2*ρ+3*(γ-1)*mx^2+(γ-1)*my^2+(γ-1)*mz^2)/ρ^2)    -((2*(Hx*Hy*ρ+(γ-1)*mx*my))/ρ^2)    -((2*(Hx*Hz*ρ+(γ-1)*mx*mz))/ρ^2)    -((2*(Hx*my+(γ-2)*Hy*mx))/ρ)    -((2*(Hx*mz+(γ-2)*Hz*mx))/ρ)
	((γ-3)*mx^2+(γ-1)*(my^2+mz^2))/(2*ρ^2)    (γ-1)/2    -(((γ-3)*mx)/ρ)    (my-γ*my)/ρ    (mz-γ*mz)/ρ    (γ-2)*(-Hy)    (γ-2)*(-Hz)
	-((mx*my)/ρ^2)    0    my/ρ    mx/ρ    0    -Hx    0
	-((mx*mz)/ρ^2)    0    mz/ρ    0    mx/ρ    0    -Hx
	(Hx*my-Hy*mx)/ρ^2    0    Hy/ρ    -(Hx/ρ)    0    mx/ρ    0
	(Hx*mz-Hz*mx)/ρ^2    0    Hz/ρ    0    -(Hx/ρ)    0    mx/ρ]
end

function w2F(w::Vector)::Matrix
	ρ = w[1]
	E = w[2]
	mx = w[3]
	my = w[4]
	mz = w[5]
	Hy = w[6]
	Hz = w[7]
	F=[mx
(-2*Hx*ρ*(Hy*my+Hz*mz)+mx*(ρ*(ρ*ρ-((ρ-2)*Hy^2)-(ρ-2)*Hz^2)-((ρ-1)*my^2)-(ρ-1)*mz^2)-((ρ-1)*mx^3))/ρ^2
1/2*(-(((ρ-1)*(-ρ*ρ+Hy^2*ρ+Hz^2*ρ+mx^2+my^2+mz^2))/ρ)+Hy^2+Hz^2+(2*mx^2)/ρ)
(mx*my)/ρ-Hx*Hy
(mx*mz)/ρ-Hx*Hz
(Hy*mx-Hx*my)/ρ
(Hz*mx-Hx*mz)/ρ]
end


function lax_wendroff(wp::Matrix, w::Matrix, C::AbstractFloat)
	for l in 2:size(w, 2)-1
		Am = 0.5*(w[:, l]+w[:, l-1]) |> w2A
		Ap = 0.5*(w[:, l]+w[:, l+1]) |> w2A
		Fm = w[:, l-1] |> w2F
		Fp = w[:, l+1] |> w2F
		F = w[:, l] |> w2F
		wp[:, l] .= w[:, l] - 0.5C*(Fp - Fm) +
		0.5C^2*(Ap*(Fp-F) - Am * (F-Fm))
	end
	wp[:, 1] .= w[:, 2] 
	wp[:, end] .= w[:, end-1] 
end

# λ=[1,2,3.]
# w = [1 2 3; 2 2 4]'
# Δw = [0, 3, 5.]

# l = 1

# %%
function init1(x::AbstractVector, U::Matrix)
	U[:, x .< 0] .= [  2.121,  4.981, -13.27, -0.163, -0.6521, 2.572, 10.29] |> w2U
	U[:, x .>= 0 ] .= [      1,      1,  -15.3,      0,       0,     1,     4] |> w2U
end

function init2(x::AbstractVector, U::Matrix)
	U[:, x .< 0] .= [  2.219, 0.4442, 0.5048, 0.0961,  0.0961,     1,     1] |> w2U
	U[:, x .>= 0 ] .= [      1,    0.1,-0.9225,      0,       0,     1,     1] |> w2U
end

function init3(x::AbstractVector, U::Matrix)
	U[:, x .< 0.2] .= [  3.896,  305.9,      0, -0.058,  -0.226, 3.951,  15.8] |> w2U
	U[:, x .>= 0.2] .= [      1,      1,  -15.3,      0,       0,     1,     4] |> w2U
end

function init4(x::AbstractVector, U::Matrix)
	U[:, x .< 0.2] .= [  3.108, 1.4336,      0, 0.2633,  0.2633,   0.1,   0.1] |> w2U
	U[:, x .>= 0.2 ] .= [      1,    0.1,-0.9225,      0,       0,     1,     1] |> w2U
end

# %%
function true_sol(x::AbstractVector, w::Matrix, t::AbstractFloat)
	a = -2.633*t
	b = -1.636*t
	c = 1.529*t
	d = 2.480*t
	y1=[0.445, 0.311, 8.928]
	y2=[0.345, 0.527, 6.570]
	w[:, x .< a] .= y1

	k=(y1-y2)./(a-b) # y = k(x-x1)+y1
	mask=@. a < x < b
	for i = 1:3
		w[i, mask] .= k[i]*(x[mask].-a) .+ y1[i]
	end

	w[:, b.< x .< c] .= y2
	w[:, c.< x .< d] .= [1.304, 1.994, 7.691]
	w[:, x .> d] .= [0.500, 0.000, 1.428]
end
# %%

struct Cells
	x::AbstractVector{Float64}
	u::Matrix{Float64} # u^n
	up::Matrix{Float64} # u^(n+1) ; u plus
	function Cells(b::Float64=-1.0, e::Float64=1.0; step::Float64=0.01, init::Function=init0)
	x = range(b, e, step=step)
	u=zeros(7,length(x))
	init(x, u)
	up=deepcopy(u)
	new(x, u , up)
	end
end

Cells(Δ::Float64)=Cells(-1.0, 1.0, step=Δ)
Cells(init::Function)=Cells(-1.0, 1.0, init=init)
Cells(b::Float64, e::Float64, Δ::Float64)=Cells(b, e, step=Δ)

next(c::Cells, flg::Bool)::Matrix = flg ? c.up : c.u
current(c::Cells, flg::Bool)::Matrix = flg ? c.u : c.up

function update!(c::Cells, flg::Bool, f::Function, C::AbstractFloat)
	UP=next(c, flg) # u^(n+1)
	U=current(c, flg) # u^n
	f(UP, U, C)
	return !flg
end
update!(c::Cells, flg::Bool, f::Function) = update!(c, flg, f, 0.5)

function f2title(f::Function)
	if f == lax_wendroff
		return "Lax-Wendroff"
	end
	return "unknown"
end


# %%
C = 0.1
Δx= 0.01
# C = Δt/Δx
Δt =  C * Δx

nx=261
# %%
function problem(C::AbstractFloat, f::Function, nx::Int = 261)

	# title = L"$m$"
	# t=0.002
	C_str=string(round(C, digits=3))
	t=0.14
	# C = 0.7/2.633
	Δx= 2/nx
	Δt = Δx * C
	# f = limiter
	f = lax_wendroff
	c=Cells(step=Δx, init=init2)
	title = f |> f2title
	# fig, ax=plt.subplots(3,1, figsize=(12,13))

	fig, ax=plt.subplots(3,1)
	fig.suptitle("t = "*string(t)*"    "*"C = "*C_str*"    "*title, fontsize=16)
	ax[1].plot(c.x, c.u[1, :], "-.k", linewidth=0.2, label=L"$\rho$(初始值)")
	ax[2].plot(c.x, c.u[2, :], "-.k", linewidth=0.2, label=L"$E$(初始值)")
	ax[3].plot(c.x, c.u[3, :], "-.k", linewidth=0.2, label=L"$ρv_x$(初始值)")
	ax[1].legend()
	ax[2].legend()
	ax[3].legend()
	plt.savefig("../figures/初值.pdf", bbox_inches="tight")
	plt.show()

	flg=true # flag
	# for _ = 1:round(Int, t/Δt)
	for _ = 1:round(Int, 15)
		flg=update!(c, flg, f, C)
	end

	w=current(c, flg)
	tw=similar(w)
	true_sol(c.x, tw, t)


	ax[1].plot(c.x, tw[1, :], linewidth=1, color="k", label=L"$\rho$(真实解)", alpha=0.5)
	ax[1].plot(c.x, w[1, :], "--b", linewidth=1, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label=L"$\rho$(数值解)")
	ax[1].set_title("密度", fontsize=14)
	ax[1].legend()
	ax[3].plot(c.x, tw[2, :], linewidth=1, color="k", label=L"$m$(真实解)", alpha=0.5)
	ax[3].plot(c.x, w[2, :], "--r", linewidth=1, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label=L"$m$(数值解)")
	ax[2].set_title("质量流", fontsize=14)
	ax[3].legend()
	ax[2].plot(c.x, tw[3, :], linewidth=1, color="k", label=L"$E$(真实解)", alpha=0.5)
	ax[2].plot(c.x, w[3, :], "--y", linewidth=1, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label=L"$E$(数值解)")
	ax[3].set_title("能量", fontsize=14)
	ax[2].legend()

	# plot(c.x, circshift(w, (0, 3)), tw)

	# # plt.plot(c.x, c.u[1, :], "-.k", linewidth=0.2, label="init")
	# w = U |> U2w

	# plt.plot(x, w[1, :], linewidth=1, color="b", label="Density")
	# plt.plot(x, w[2, :], linewidth=1, color="r", label="m")
	# plt.plot(x, w[3, :], linewidth=1, color="y", label="E")
	# # plt.plot(c.x, c.u[1, :], "-.k", linewidth=0.2, label="init")
	# plt.show()

	# plt.title("time = "*string(t)*", "*"C = "*string(C)*", "* title )
	# # plt.plot(c.x, c.up, linestyle="dashed", linewidth=0.4, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label="up")
	plt.savefig("../figures/"*string(f)*string(nx)*".pdf", bbox_inches="tight")
	# plt.show()

end
# %%
function problem_non(C::AbstractFloat, f::Function, nx::Int = 261)

	# title = L"$m$"
	# t=0.002
	C_str=string(round(C, digits=3))
	t=0.28
	C = C/4.694
	# C = 0.7/2.633
	Δx= 2/nx
	Δt = Δx * C
	# f = limiter
	c=Cells(step=Δx, init=init_non)
	title = f |> f2title
	fig, ax=plt.subplots(3,1, figsize=(12,13))
	fig.suptitle("t = "*string(t)*"    "*"C = "*C_str*"    "*title, fontsize=16)
	ax[1].plot(c.x, c.u[1, :], "-.k", linewidth=0.2, label=L"$\rho$(初始值)")
	ax[3].plot(c.x, c.u[2, :], "-.k", linewidth=0.2, label=L"$m$(初始值)")
	ax[2].plot(c.x, c.u[3, :], "-.k", linewidth=0.2, label=L"$E$(初始值)")

	flg=true # flag
	for _ = 1:round(Int, t/Δt)
		flg=update!(c, flg, f, C)
	end
	U=current(c, flg)
	w = U |> U2w
	tw=similar(w)
	true_sol(c.x, tw, t)


	ax[1].plot(c.x, tw[1, :], linewidth=1, color="k", label=L"$\rho$(真实解)", alpha=0.5)
	ax[1].plot(c.x, w[1, :], "--b", linewidth=1, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label=L"$\rho$(数值解)")
	ax[1].set_title("密度", fontsize=14)
	ax[1].legend()
	ax[3].plot(c.x, tw[2, :], linewidth=1, color="k", label=L"$m$(真实解)", alpha=0.5)
	ax[3].plot(c.x, w[2, :], "--r", linewidth=1, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label=L"$m$(数值解)")
	ax[2].set_title("质量流", fontsize=14)
	ax[3].legend()
	ax[2].plot(c.x, tw[3, :], linewidth=1, color="k", label=L"$E$(真实解)", alpha=0.5)
	ax[2].plot(c.x, w[3, :], "--y", linewidth=1, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label=L"$E$(数值解)")
	ax[3].set_title("能量", fontsize=14)
	ax[2].legend()

	# plot(c.x, circshift(w, (0, 3)), tw)

	# # plt.plot(c.x, c.u[1, :], "-.k", linewidth=0.2, label="init")
	# w = U |> U2w

	# plt.plot(x, w[1, :], linewidth=1, color="b", label="Density")
	# plt.plot(x, w[2, :], linewidth=1, color="r", label="m")
	# plt.plot(x, w[3, :], linewidth=1, color="y", label="E")
	# # plt.plot(c.x, c.u[1, :], "-.k", linewidth=0.2, label="init")
	# plt.show()

	# plt.title("time = "*string(t)*", "*"C = "*string(C)*", "* title )
	# # plt.plot(c.x, c.up, linestyle="dashed", linewidth=0.4, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label="up")
	plt.savefig("../figures/"*string(f)*string(nx)*".pdf", bbox_inches="tight")
	# plt.show()

end
# %%

problem(0.18, limiter)
# problem1(0.18, limiter)
plt.show()

problem(0.05, limiter, 133)
plt.show()

problem(0.5, lax_wendroff)
plt.show()

problem(0.18, NND)
plt.show()


problem_non(0.5, upwind_non)
plt.show()

# problem1(0.5, upwind)

problem(0.5, upwind)
plt.show()

# %%

function main()
	problem(0.05, limiter)
	# problem1(0.18, limiter)
	problem(0.05, limiter, 133)
	problem(0.5, lax_wendroff)
	problem_non(0.1, upwind_non)
	problem(0.5, upwind)
	plt.show()
end
main()
