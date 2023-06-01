#=
	Created On  : 2023-04-03 00:24
    Copyright © 2023 Junyi Xu <jyxu@mail.ustc.edu.cn>

    Distributed under terms of the MIT license.
=#

using PyCall
using LinearAlgebra

# %%
@pyimport matplotlib.pyplot as plt
@pyimport matplotlib # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
matplotlib.rc("font", size=10)

const γ = 5/3
const β = 2.0
const Hx = 5.0

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

function U2A(U::Vector)::Matrix
	ρ = U[1]
	E = U[2]
	mx = U[3]
	my = U[4]
	mz = U[5]
	Hy = U[6]
	Hz = U[7]
	A = [0    0    1    0    0    0    0
	(2*Hx*ρ*(Hy*my+Hz*mz)+mx*(ρ*(-γ*E+(γ-2)*Hy^2+(γ-2)*Hz^2)+2*(γ-1)*my^2+2*(γ-1)*mz^2)+2*(γ-1)*mx^3)/ρ^3    (γ*mx)/ρ    -((-γ*E*ρ+(γ-2)*Hy^2*ρ+(γ-2)*Hz^2*ρ+3*(γ-1)*mx^2+(γ-1)*my^2+(γ-1)*mz^2)/ρ^2)    -((2*(Hx*Hy*ρ+(γ-1)*mx*my))/ρ^2)    -((2*(Hx*Hz*ρ+(γ-1)*mx*mz))/ρ^2)    -((2*(Hx*my+(γ-2)*Hy*mx))/ρ)    -((2*(Hx*mz+(γ-2)*Hz*mx))/ρ)
	((γ-3)*mx^2+(γ-1)*(my^2+mz^2))/(2*ρ^2)    (γ-1)/2    -(((γ-3)*mx)/ρ)    (my-γ*my)/ρ    (mz-γ*mz)/ρ    (γ-2)*(-Hy)    (γ-2)*(-Hz)
	-((mx*my)/ρ^2)    0    my/ρ    mx/ρ    0    -Hx    0
	-((mx*mz)/ρ^2)    0    mz/ρ    0    mx/ρ    0    -Hx
	(Hx*my-Hy*mx)/ρ^2    0    Hy/ρ    -(Hx/ρ)    0    mx/ρ    0
	(Hx*mz-Hz*mx)/ρ^2    0    Hz/ρ    0    -(Hx/ρ)    0    mx/ρ]
end

function U2F(U::Vector)::Vector
	ρ = U[1]
	E = U[2]
	mx = U[3]
	my = U[4]
	mz = U[5]
	Hy = U[6]
	Hz = U[7]
	F=[mx
(-2*Hx*ρ*(Hy*my+Hz*mz)+mx*(ρ*(γ*E-((γ-2)*Hy^2)-(γ-2)*Hz^2)-((γ-1)*my^2)-(γ-1)*mz^2)-((γ-1)*mx^3))/ρ^2
1/2*(-(((γ-1)*(-E*ρ+Hy^2*ρ+Hz^2*ρ+mx^2+my^2+mz^2))/ρ)+Hy^2+Hz^2+(2*mx^2)/ρ)
(mx*my)/ρ-Hx*Hy
(mx*mz)/ρ-Hx*Hz
(Hy*mx-Hx*my)/ρ
(Hz*mx-Hx*mz)/ρ]
end


function lax_wendroff(UP::Matrix, U::Matrix, C::AbstractFloat)
	for l in 2:size(U, 2)-1
	Am = 0.5*(U[:, l]+U[:, l-1]) |> U2A
	Ap = 0.5*(U[:, l]+U[:, l+1]) |> U2A
	Fm = U[:, l-1] |> U2F
	Fp = U[:, l+1] |> U2F
	F = U[:, l] |> U2F
	UP[:, l] .= U[:, l] - 0.5C*(Fp - Fm) +
	0.5C^2*(Ap*(Fp-F) - Am * (F-Fm))
	# println(0.5C^2*(Ap*(Fp-F) - Am * (F-Fm)))
	# UP[:, l] .= U[:, l] - 0.5C*(Fp - Fm)
	end
	UP[:, 1] .= U[:, 2] 
	UP[:, end] .= U[:, end-1] 
end

l = 135
function upwind(UP::Matrix, U::Matrix, C::AbstractFloat)
for l in 2:size(U, 2)-1
    # Rm, λm, Lm = U[:, l] |> U2A |> svd
    R, λ, L = U[:, l] |> U2A |> svd
    for k = 1:7
        Σ=0.0
        for i = 1:7
            s = λ[i] >= 0 ? 1 : -1
            # Rp, λp, Lp = U[:, l-s] |> U2A |> svd
            # Rm, λm, Lm = U[:, l] |> svd
            # R = 0.5*(Rm+Rp)
            # Λ = 0.5(λ+λp)
            # L = 0.5*(Lm+Lp)
            for j = 1:7
                a = s*λ[i]*R[k, i]*L[i, j]
                Σ += a*(U[j, l] - U[j, l-s])
            end
        end
        UP[k, l] =U[k, l] - C*Σ
    end
end
# UP[:, 1] .= U[:, 2] 
# UP[:, end] .= U[:, end-1] 
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

function main(C::AbstractFloat, init::Function, nx::Int = 261)
	ylabels=["\$\\rho\$"
			"\$E\$"
			"\$ρv_x\$"
			"\$ρv_y\$"
			"\$ρv_z\$"
			"\$H_y\$"
			"\$H_z\$"]

	C_str=string(round(C, digits=3))
	t=0.3
	Δx= 2/nx
	Δt = Δx * C

	f = lax_wendroff
	title = f |> f2title

	c=Cells(step=Δx, init=init)
	fig, ax=plt.subplots(7,1, figsize=(6, 7))
	# fig.suptitle("t = "*string(t)*"    "*"C = "*C_str*"    "*title, fontsize=9)
	# fig.suptitle("t = "*string(t)*"    "*"C = "*C_str*"    "*title)
	for i = 1:7
		ax[i].plot(c.x, c.u[i, :], "-.k", linewidth=1)
	end
	flg=true # flag
	N = round(Int, t/Δt)
	# N = round(Int, 10)
	for n = 1:N
		flg=update!(c, flg, f, C)
		################ plot ######################
		if n == round(Int, N/3) || n == round(Int, 2*N/3) || n == N
			for i = 1:7
				ax[i].plot(c.x, c.u[i, :], linewidth=1, marker="o", markerfacecolor="none", markeredgewidth=0.3, markersize=2, label="t = $(round(n * Δt, digits=2))s")
				ax[i].legend()
				ax[i].set_ylabel(ylabels[i]) 
			end
			for i = 1:6
				ax[i].set_xticks([]) # 只有最后一个有 x 轴
			end
		end ################ end plot #############
	end
	plt.savefig("../figures/"*string(init)*".pdf", bbox_inches="tight")
	plt.show()
end


main(0.1, init4)

# f=[6,24,322]
# A=Tridiagonal([1,1],[6,4,14], [1,1])

# @time begin 
# A \ f 
# end
